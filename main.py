import os
import argparse
import pathspec
import hashlib
from typing import Dict, List, Optional
from tqdm import tqdm
from openai import OpenAI
from halo import Halo
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from colorama import Fore, Style, init
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, ConfigDict, PrivateAttr


init(autoreset=True)


EXT_TO_LANG: Dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".hpp": Language.CPP,
    ".cc": Language.CPP,
    ".cs": Language.CSHARP,
    ".go": Language.GO,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".swift": Language.SWIFT,
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    ".php": Language.PHP,
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
    ".rst": Language.RST,
    ".tex": Language.LATEX,
    ".sol": Language.SOL,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".ps1": Language.POWERSHELL,
    ".r": Language.R,
    ".proto": Language.PROTO,
    ".hs": Language.HASKELL,
    ".cbl": Language.COBOL,
    ".vb": Language.VISUALBASIC6,
}


class Settings(BaseSettings):
    qdrant_url: str
    base_llm_provider_url: str
    llm_provider_api_key: str
    embd_model: str
    embd_vector_size: int

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class EmbeddingProvider(BaseModel):
    api_key: str
    base_url: str
    embd_model: str

    _client: OpenAI = PrivateAttr()

    def model_post_init(self) -> None:
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        response = self._client.embeddings.create(input=[text], model=self.embd_model)
        return response.data[0].embedding


class App(BaseModel):
    project_path: str
    qdrant_url: str
    base_llm_provider_url: str
    llm_provider_api_key: str
    embd_model: str
    embd_vector_size: int
    collection_name: Optional[str] = None

    _client: QdrantClient = PrivateAttr()
    _provider: EmbeddingProvider = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self) -> None:
        self._client = QdrantClient(self.qdrant_url)
        self._provider = EmbeddingProvider(
            api_key=self.llm_provider_api_key,
            base_url=self.base_llm_provider_url,
            embd_model=self.embd_model,
        )

        if not self.collection_name:
            folder_name = os.path.basename(os.path.abspath(self.project_path))
            self.collection_name = (
                f"{folder_name.lower().replace('-', '_').replace(' ', '_')}_codebase"
            )

    def _get_gitignore_spec(self) -> pathspec.PathSpec:
        gitignore_path = os.path.join(self.project_path, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r", encoding="utf-8") as f:
                return pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
        return pathspec.PathSpec.from_lines(
            "gitwildmatch", ["node_modules/", "dist/", ".git/"]
        )

    def _init_collection(self) -> None:
        spinner = Halo(
            text=f" Проверка коллекции {self.collection_name}...", spinner="dots"
        ).start()
        try:
            if not self._client.collection_exists(self.collection_name):
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embd_vector_size, distance=Distance.COSINE
                    ),
                )
                self._client.create_payload_index(
                    self.collection_name, "path", models.PayloadSchemaType.KEYWORD
                )
                self._client.create_payload_index(
                    self.collection_name, "hash", models.PayloadSchemaType.KEYWORD
                )
                spinner.succeed(f" Коллекция {self.collection_name} создана.")
            else:
                spinner.info(
                    f" Используется существующая коллекция {self.collection_name}."
                )
        except Exception as e:
            spinner.fail(f" Ошибка базы данных: {e}")
            exit(1)

    def _index_files(self) -> None:
        project_path = os.path.abspath(self.project_path)
        spec = self._get_gitignore_spec()

        self._init_collection()

        scan_spinner = Halo(
            text=" Сканирование файлов проекта...", spinner="dots"
        ).start()

        files_to_index: List[str] = []
        for root, dirs, files in os.walk(project_path):
            rel_root = os.path.relpath(root, project_path)
            dirs[:] = [
                d for d in dirs if not spec.match_file(os.path.join(rel_root, d))
            ]

            for file in files:
                rel_file_path = os.path.join(rel_root, file)
                if (
                    not spec.match_file(rel_file_path)
                    and os.path.splitext(file)[1].lower() in EXT_TO_LANG
                ):
                    files_to_index.append(os.path.join(root, file))

        if not files_to_index:
            scan_spinner.warn(" Подходящих файлов не найдено.")
            return

        scan_spinner.succeed(f" Файлов для анализа: {len(files_to_index)}")

        skipped_count = 0
        indexed_count = 0

        with tqdm(
            total=len(files_to_index),
            desc="🗂️ Подготовка",
            unit="file",
            colour="green",
            dynamic_ncols=True,
        ) as progress_bar:
            for path in files_to_index:
                file_name = os.path.basename(path)
                progress_bar.set_postfix(
                    {"file": f"{Fore.CYAN}{file_name[:15]}{Style.RESET_ALL}"}
                )

                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        code = f.read()

                    current_hash = hashlib.md5(code.encode()).hexdigest()

                    search_result = self._client.scroll(
                        collection_name=self.collection_name,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="path", match=models.MatchValue(value=path)
                                ),
                                models.FieldCondition(
                                    key="hash",
                                    match=models.MatchValue(value=current_hash),
                                ),
                            ]
                        ),
                        limit=1,
                        with_payload=False,
                        with_vectors=False,
                    )[0]

                    if search_result:
                        skipped_count += 1
                        progress_bar.set_description(f"🗂️ Пропущено: {skipped_count}")
                        progress_bar.update(1)
                        continue

                    indexed_count += 1
                    progress_bar.set_description(f"🚀 Индексация: {indexed_count}")

                    ext = os.path.splitext(path)[1].lower()
                    lang = EXT_TO_LANG[ext]

                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=lang, chunk_size=1200, chunk_overlap=150
                    )
                    chunks = splitter.split_text(code)

                    response = self._provider._client.embeddings.create(
                        input=chunks, model=self._provider.embd_model
                    )
                    vectors = [item.embedding for item in response.data]

                    points: List[PointStruct] = []
                    for i, chunk in enumerate(chunks):
                        unique_id_str = f"{path}_{i}"
                        point_id = (
                            int(hashlib.md5(unique_id_str.encode()).hexdigest(), 16)
                            % 10**15
                        )

                        points.append(
                            PointStruct(
                                id=point_id,
                                vector=vectors[i],
                                payload={
                                    "path": path,
                                    "content": f"File: {file_name}\n\n{chunk}",
                                    "language": str(lang),
                                    "hash": current_hash,
                                },
                            )
                        )

                    self._client.upsert(
                        collection_name=self.collection_name, points=points
                    )
                    progress_bar.update(1)

                except Exception as e:
                    progress_bar.write(
                        f"{Fore.RED}❌ Ошибка в {file_name}: {e}{Style.RESET_ALL}"
                    )
                    progress_bar.update(1)

        print(f"\n{Fore.GREEN}✅ Синхронизация завершена!{Style.RESET_ALL}")
        print(f"📦 Коллекция: {self.collection_name}")
        print(f"💽 Обновлено/Добавлено: {indexed_count}")
        print(f"⏩ Пропущено без изменений: {skipped_count}")

    def run(self) -> None:
        self._index_files()


if __name__ == "__main__":
    config = Settings()
    parser = argparse.ArgumentParser(description="Умная индексация кода в Qdrant")
    parser.add_argument("project_path", help="Путь к папке проекта")
    parser.add_argument("--collection_name", help="Имя коллекции")
    parser.add_argument(
        "--embd_model", default=config.embd_model, help="Имя модели эмбеддинга"
    )
    parser.add_argument(
        "--embd_vector_size",
        type=int,
        default=config.embd_vector_size,
        help="Размер вектора",
    )
    parser.add_argument("--qdrant_url", default=config.qdrant_url, help="URL Qdrant")
    parser.add_argument(
        "--base_llm_provider_url",
        default=config.base_llm_provider_url,
        help="URL провайдера LLM",
    )
    parser.add_argument(
        "--llm_provider_api_key",
        default=config.llm_provider_api_key,
        help="API ключ провайдера LLM",
    )

    args = parser.parse_args()

    if bool(args.base_llm_provider_url) != bool(args.llm_provider_api_key):
        parser.error("Аргументы URL провайдера и API ключа должны быть заданы вместе.")
    if bool(args.embd_model) != bool(args.embd_vector_size):
        parser.error("Аргументы модели и размера вектора должны быть заданы вместе.")

    init_kwargs = {
        k: v for k, v in vars(args).items() if v is not None and k != "project_path"
    }

    app = App(project_path=args.project_path, **init_kwargs)
    app.run()
