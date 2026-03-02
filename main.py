import os
import argparse
import pathspec
from tqdm import tqdm
from openai import OpenAI
from halo import Halo
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


EXT_TO_LANG = {
    # Python
    ".py": Language.PYTHON,
    # JavaScript / TypeScript / React
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    # C / C++ / C#
    ".c": Language.C,
    ".cpp": Language.CPP,
    ".hpp": Language.CPP,
    ".cc": Language.CPP,
    ".cs": Language.CSHARP,
    # Go
    ".go": Language.GO,
    # Java / Kotlin / Scala
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".scala": Language.SCALA,
    # Ruby / Rust / Swift / Elixir
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".swift": Language.SWIFT,
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    # PHP
    ".php": Language.PHP,
    # Web (HTML / Markdown / RST / LaTeX)
    ".html": Language.HTML,
    ".md": Language.MARKDOWN,
    ".rst": Language.RST,
    ".tex": Language.LATEX,
    # SQL / Solidity
    ".sol": Language.SOL,
    # Shell / Scripting
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".ps1": Language.POWERSHELL,
    # Data / Logic / Protobuf
    ".r": Language.R,
    ".proto": Language.PROTO,
    ".hs": Language.HASKELL,
    # Legacy / Enterprise
    ".cbl": Language.COBOL,
    ".vb": Language.VISUALBASIC6,
}


class EmbeddingProvider:
    def __init__(self, api_key: str, base_url: str, embd_model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.embd_model = embd_model

    def get_embedding(self, text: str):
        text = text.replace("\n", " ")

        response = self.client.embeddings.create(input=[text], model=self.embd_model)
        return response.data[0].embedding


class App:
    def __init__(
        self,
        project_path: str,
        collection_name: str | None = None,
        qdrant_url: str = "http://localhost:6333",
        base_llm_provider_url: str = "http://localhost:1234/v1",
        llm_provider_api_key: str = "not-needed",
        embd_model: str = "text-embedding-nomic-embed-text-v1.5",
        embd_vector_size: int = 768,
    ):
        self.client = QdrantClient(qdrant_url)
        self.provider = EmbeddingProvider(
            api_key=llm_provider_api_key,
            base_url=base_llm_provider_url,
            embd_model=embd_model,
        )
        self.embd_vector_size = embd_vector_size
        self.project_path = project_path
        if not collection_name:
            folder_name = os.path.basename(os.path.abspath(project_path))
            collection_name = (
                f"{folder_name.lower().replace('-', '_').replace(' ', '_')}_codebase"
            )

        self.collection_name = collection_name

    def _get_gitignore_spec(self):
        """Внутренний метод для загрузки правил игнорирования"""
        gitignore_path = os.path.join(self.project_path, ".gitignore")
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r", encoding="utf-8") as f:
                return pathspec.PathSpec.from_lines("gitwildmatch", f.readlines())
        return pathspec.PathSpec.from_lines(
            "gitwildmatch", ["node_modules/", "dist/", ".git/"]
        )

    def _init_collection(self):
        """Инициализация коллекции в Qdrant с использованием self.client"""
        spinner = Halo(
            text=f" Проверка коллекции {self.collection_name}...", spinner="dots"
        ).start()
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embd_vector_size, distance=Distance.COSINE
                    ),
                )
                spinner.succeed(f" Коллекция {self.collection_name} создана.")
            else:
                spinner.info(
                    f" Используется существующая коллекция {self.collection_name}."
                )
        except Exception as e:
            spinner.fail(f" Ошибка базы данных: {e}")
            exit(1)

    def _index_files(self):
        """Внутренний метод индексации, использующий состояние объекта App"""
        project_path = os.path.abspath(self.project_path)
        spec = self._get_gitignore_spec()

        self._init_collection()

        files_to_index = []
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
            print(" Подходящих файлов для индексации не найдено.")
            return

        with tqdm(
            total=len(files_to_index),
            desc=" Индексация проекта",
            unit="file",
            colour="green",
        ) as progress_bar:
            for path in files_to_index:
                try:
                    ext = os.path.splitext(path)[1].lower()
                    lang = EXT_TO_LANG[ext]

                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        code = f.read()

                    splitter = RecursiveCharacterTextSplitter.from_language(
                        language=lang, chunk_size=1200, chunk_overlap=150
                    )
                    chunks = splitter.split_text(code)

                    points = []
                    for i, chunk in enumerate(chunks):
                        vector = self.provider.get_embedding(chunk)
                        points.append(
                            PointStruct(
                                id=hash(f"{path}_{i}") % 10**10,
                                vector=vector,
                                payload={
                                    "path": path,
                                    "content": chunk,
                                    "language": str(lang),
                                },
                            )
                        )

                    if points:
                        self.client.upsert(
                            collection_name=self.collection_name, points=points
                        )

                    progress_bar.update(1)
                except Exception as e:
                    tqdm.write(f" Ошибка в файле {path}: {e}")
                    progress_bar.update(1)

    def run(self):
        """Теперь запуск максимально прост"""
        self._index_files()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Умная индексация кода в Qdrant")
    parser.add_argument("project_path", help="Путь к папке проекта")

    parser.add_argument("--collection_name", help="Имя коллекции")
    parser.add_argument("--embd_model", help="Имя модели эмбеддинга")
    parser.add_argument("--qdrant_url", help="URL Qdrant")
    parser.add_argument("--base_llm_provider_url", help="URL провайдера (LM Studio)")

    args = parser.parse_args()

    init_kwargs = {
        k: v for k, v in vars(args).items() if v is not None and k != "project_path"
    }

    app = App(**init_kwargs, project_path=args.project_path)
    app.run()
