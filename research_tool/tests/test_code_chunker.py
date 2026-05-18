"""Tests for research_tool.code_chunker — AST-aware code chunking."""

import pytest

from research_tool.code_chunker import (
    CodeChunk,
    chunk_code_file,
    detect_language,
    _TREE_SITTER_AVAILABLE,
    _estimate_tokens,
    _SMALL_CLASS_TOKENS,
    _MIN_CHUNK_CHARS,
    _merge_small_chunks,
)


class TestDetectLanguage:
    def test_csharp_extension(self):
        assert detect_language("src/MyClass.cs") == "c_sharp"

    def test_python_extension(self):
        assert detect_language("utils/helper.py") == "python"

    def test_unknown_extension(self):
        assert detect_language("data/file.xyz") is None

    def test_case_insensitive(self):
        assert detect_language("Main.CS") == "c_sharp"

    @pytest.mark.parametrize("ext,expected", [
        (".py", "python"), (".pyi", "python"), (".pyw", "python"),
        (".go", "go"),
        (".java", "java"), (".kt", "java"), (".groovy", "java"),
        (".js", "javascript"), (".jsx", "javascript"), (".mjs", "javascript"),
        (".ts", "typescript"), (".tsx", "typescript"),
        (".sh", "bash"), (".bash", "bash"), (".zsh", "bash"),
        (".html", "html"), (".htm", "html"), (".vue", "html"),
        (".c", "c"), (".h", "c"),
        (".cpp", "cpp"), (".cc", "cpp"), (".hpp", "cpp"),
        (".rs", "rust"),
        (".rb", "ruby"), (".rake", "ruby"),
        # Project / build files
        (".csproj", "xml"), (".fsproj", "xml"), (".vbproj", "xml"),
        (".sln", "xml"), (".props", "xml"), (".targets", "xml"), (".nuspec", "xml"),
        (".xml", "xml"), (".xsl", "xml"), (".svg", "xml"), (".plist", "xml"),
        (".xaml", "xml"), (".resx", "xml"),
        # CSS
        (".css", "css"), (".scss", "css"), (".less", "css"),
        # SQL
        (".sql", "sql"),
        # Config
        (".tf", "hcl"), (".tfvars", "hcl"),
        (".yaml", "yaml"), (".yml", "yaml"),
        (".toml", "toml"),
        (".json", "json"), (".jsonc", "json"),
    ])
    def test_all_extensions(self, ext, expected):
        assert detect_language(f"src/file{ext}") == expected

    @pytest.mark.parametrize("filename,expected", [
        ("Gemfile", "ruby"),
        ("Rakefile", "ruby"),
        ("Vagrantfile", "ruby"),
        ("Pipfile", "toml"),
        ("Makefile", "bash"),
        ("Jenkinsfile", "java"),
    ])
    def test_filename_detection(self, filename, expected):
        assert detect_language(filename) == expected


class TestChunkCodeFilePython:
    """Test Python code chunking with tree-sitter."""

    @pytest.fixture
    def python_3_methods(self):
        return '''
def hello():
    """Say hello."""
    print("hello")

def goodbye():
    """Say goodbye."""
    print("goodbye")

def calculate(x, y):
    """Add two numbers."""
    return x + y
'''

    @pytest.fixture
    def python_small_class(self):
        return '''
class SmallHelper:
    """A small helper class."""

    def __init__(self):
        self.x = 1

    def get_x(self):
        return self.x
'''

    @pytest.fixture
    def python_large_class(self):
        methods = "\n".join(
            f"    def method_{i}(self, value):\n        \"\"\"Process value {i} with detailed logic.\"\"\"\n        result = value * {i} + {i * 2}\n        if result > 100:\n            result = result // 2\n        return result\n"
            for i in range(15)
        )
        return f"class BigClass:\n{methods}"

    def test_three_functions_produce_three_chunks(self, python_3_methods):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("test.py", python_3_methods, "python")
        assert len(chunks) == 3
        assert all(c.content_type == "code" for c in chunks)
        assert all(c.language == "python" for c in chunks)

    def test_chunks_contain_metadata_enrichment(self, python_3_methods):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("src/utils.py", python_3_methods, "python")
        assert chunks[0].text.startswith("// File: src/utils.py")

    def test_small_class_single_chunk(self, python_small_class):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("helper.py", python_small_class, "python")
        # Small class should be chunked as a single unit
        assert len(chunks) == 1
        assert "SmallHelper" in chunks[0].text

    def test_large_class_method_level_chunks(self, python_large_class):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("big.py", python_large_class, "python")
        # Large class with 15 methods → chunks at method level
        assert len(chunks) == 15
        assert all("method_" in c.text for c in chunks)

    def test_empty_file_no_chunks(self):
        chunks = chunk_code_file("empty.py", "", "python")
        assert chunks == []

    def test_whitespace_only_no_chunks(self):
        chunks = chunk_code_file("blank.py", "   \n\n   ", "python")
        assert chunks == []


class TestChunkCodeFileCSharp:
    """Test C# code chunking."""

    @pytest.fixture
    def csharp_with_methods(self):
        return '''
namespace MyApp
{
    public class Calculator
    {
        public int Add(int a, int b)
        {
            return a + b;
        }

        public int Subtract(int a, int b)
        {
            return a - b;
        }

        public int Multiply(int a, int b)
        {
            return a * b;
        }
    }
}
'''

    def test_small_csharp_class_single_chunk(self, csharp_with_methods):
        """Small C# class (<500 tokens) should produce 1 chunk at class level."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("Calculator.cs", csharp_with_methods, "c_sharp")
        assert len(chunks) == 1
        assert chunks[0].content_type == "code"
        assert "Calculator" in chunks[0].metadata.get("class", "")
        assert "Add" in chunks[0].text
        assert "Subtract" in chunks[0].text

    def test_large_csharp_class_method_level(self):
        """Large C# class (>500 tokens) should produce method-level chunks."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        methods = "\n".join(
            f"        public int Method{i}(int a, int b)\n        {{\n            // Implementation of method {i} with enough content to be meaningful\n            var result = a + b + {i};\n            return result;\n        }}\n"
            for i in range(12)
        )
        source = f"namespace MyApp {{\n    public class BigCalc {{\n{methods}    }}\n}}"
        chunks = chunk_code_file("BigCalc.cs", source, "c_sharp")
        assert len(chunks) == 12
        assert all(c.content_type == "code" for c in chunks)

    def test_csharp_metadata_has_namespace(self, csharp_with_methods):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("Calculator.cs", csharp_with_methods, "c_sharp")
        # Class-level chunk should have namespace as parent context
        assert "MyApp" in chunks[0].metadata.get("namespace", "")


class TestCSharpRecordChunking:
    """Test C# record declarations are chunked as single units."""

    @pytest.fixture
    def csharp_record(self):
        return '''
namespace MyApp
{
    public record RetryPolicy
    {
        public int MaxRetries { get; init; } = 3;
        public TimeSpan BaseDelay { get; init; } = TimeSpan.FromMilliseconds(100);
        public TimeSpan MaxDelay { get; init; } = TimeSpan.FromSeconds(30);
        public int JitterMs { get; init; } = 50;
    }
}
'''

    def test_small_record_produces_single_chunk(self, csharp_record):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("Policy.cs", csharp_record, "c_sharp")
        assert len(chunks) == 1
        assert "RetryPolicy" in chunks[0].text
        assert "MaxRetries" in chunks[0].text
        assert "MaxDelay" in chunks[0].text

    def test_record_metadata_has_name(self, csharp_record):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("Policy.cs", csharp_record, "c_sharp")
        assert chunks[0].metadata.get("class") == "RetryPolicy"
        assert "MyApp" in chunks[0].metadata.get("namespace", "")

    def test_large_record_splits_at_member_level(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        props = "\n".join(
            f"        public string Prop{i} {{ get; init; }} = \"value{i}\";"
            for i in range(40)
        )
        source = f"namespace MyApp {{\n    public record BigRecord {{\n{props}\n    }}\n}}"
        chunks = chunk_code_file("Big.cs", source, "c_sharp")
        assert len(chunks) > 1


class TestCSharpEnumChunking:
    """Test C# enum declarations produce chunks."""

    @pytest.fixture
    def csharp_enum(self):
        return '''
namespace MyApp
{
    public enum DeliveryMode
    {
        FireAndForget = 0,
        AtLeastOnce = 1,
        ExactlyOnce = 2
    }
}
'''

    def test_enum_produces_single_chunk(self, csharp_enum):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("Modes.cs", csharp_enum, "c_sharp")
        assert len(chunks) == 1
        assert "DeliveryMode" in chunks[0].text
        assert "FireAndForget" in chunks[0].text

    def test_enum_metadata_has_name(self, csharp_enum):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("Modes.cs", csharp_enum, "c_sharp")
        assert chunks[0].metadata.get("class") == "DeliveryMode"


class TestJavaEnumChunking:
    """Test Java enum declarations produce chunks."""

    def test_java_enum_produces_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''public enum Priority {
    LOW,
    NORMAL,
    HIGH,
    CRITICAL
}
'''
        chunks = chunk_code_file("Priority.java", source, "java")
        assert len(chunks) == 1
        assert "Priority" in chunks[0].text
        assert "CRITICAL" in chunks[0].text


class TestRustEnumChunking:
    """Test Rust enum declarations produce chunks."""

    def test_rust_enum_produces_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''enum Color {
    Red,
    Green,
    Blue,
    Custom(u8, u8, u8),
}
'''
        chunks = chunk_code_file("color.rs", source, "rust")
        assert len(chunks) == 1
        assert "Color" in chunks[0].text
        assert "Custom" in chunks[0].text


class TestTypeScriptEnumChunking:
    """Test TypeScript enum declarations produce chunks."""

    def test_ts_enum_produces_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''enum Direction {
    Up = "UP",
    Down = "DOWN",
    Left = "LEFT",
    Right = "RIGHT",
}
'''
        chunks = chunk_code_file("direction.ts", source, "typescript")
        assert len(chunks) == 1
        assert "Direction" in chunks[0].text
        assert "UP" in chunks[0].text


class TestMergeSmallChunks:
    """Test _merge_small_chunks post-processing."""

    def _make_chunk(self, text: str, namespace: str = "MyApp",
                    cls: str = "", func: str = "") -> CodeChunk:
        meta: dict = {"namespace": namespace}
        if cls:
            meta["class"] = cls
        if func:
            meta["function"] = func
        return CodeChunk(text=text, file_path="test.cs", language="c_sharp", metadata=meta)

    def test_empty_list_returns_empty(self):
        assert _merge_small_chunks([]) == []

    def test_two_small_same_namespace_merge(self):
        a = self._make_chunk("x" * 100, cls="EnumA")
        b = self._make_chunk("y" * 100, cls="EnumB")
        result = _merge_small_chunks([a, b])
        assert len(result) == 1
        assert "x" * 100 in result[0].text
        assert "y" * 100 in result[0].text
        assert result[0].metadata["class"] == "EnumA, EnumB"

    def test_three_small_same_namespace_merge(self):
        a = self._make_chunk("a" * 80, cls="A")
        b = self._make_chunk("b" * 80, cls="B")
        c = self._make_chunk("c" * 80, cls="C")
        result = _merge_small_chunks([a, b, c])
        assert len(result) == 1
        assert result[0].metadata["class"] == "A, B, C"

    def test_different_namespace_no_merge(self):
        a = self._make_chunk("x" * 100, namespace="Ns1", cls="A")
        b = self._make_chunk("y" * 100, namespace="Ns2", cls="B")
        result = _merge_small_chunks([a, b])
        assert len(result) == 2

    def test_empty_namespace_no_merge(self):
        """Top-level chunks with empty namespace should not merge."""
        a = self._make_chunk("x" * 100, namespace="", func="hello")
        b = self._make_chunk("y" * 100, namespace="", func="goodbye")
        result = _merge_small_chunks([a, b])
        assert len(result) == 2

    def test_small_merged_into_large_same_type(self):
        """Small type-level chunk merges into adjacent large type-level chunk."""
        big = self._make_chunk("x" * 300, cls="Big")
        small = self._make_chunk("y" * 100, cls="Small")
        result = _merge_small_chunks([big, small])
        assert len(result) == 1
        assert "Small" in result[0].metadata.get("class", "")

    def test_orphan_small_chunk_stays(self):
        small = self._make_chunk("x" * 100, namespace="Alone", cls="Solo")
        result = _merge_small_chunks([small])
        assert len(result) == 1
        assert result[0].metadata["class"] == "Solo"

    def test_exactly_at_threshold_merges_with_small(self):
        """Chunk at threshold merges with adjacent small type-level chunk."""
        at_threshold = self._make_chunk("x" * _MIN_CHUNK_CHARS, cls="Exact")
        small = self._make_chunk("y" * 100, cls="Small")
        result = _merge_small_chunks([at_threshold, small])
        assert len(result) == 1
        assert "Exact" in result[0].metadata.get("class", "")

    def test_small_before_large_merges(self):
        small = self._make_chunk("x" * 100, cls="Small")
        big = self._make_chunk("y" * 300, cls="Big")
        result = _merge_small_chunks([small, big])
        assert len(result) == 1
        assert "Small" in result[0].metadata.get("class", "")
        assert "Big" in result[0].metadata.get("class", "")

    def test_function_level_chunks_not_merged(self):
        """Method/function chunks should not merge even when small."""
        a = self._make_chunk("x" * 100, func="hello")
        b = self._make_chunk("y" * 100, func="goodbye")
        result = _merge_small_chunks([a, b])
        assert len(result) == 2

    def test_merge_separator_is_double_newline(self):
        a = self._make_chunk("AAA", cls="A")
        b = self._make_chunk("BBB", cls="B")
        result = _merge_small_chunks([a, b])
        assert "\n\n" in result[0].text

    def test_integration_csharp_corpus_no_mergeable_micro_chunks(self):
        """C# corpus should have no micro-chunks with same-namespace neighbors after merge."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        import pathlib
        corpus_path = pathlib.Path(__file__).parent.parent / "benchmarks" / "corpus" / "sample_csharp.cs"
        if not corpus_path.exists():
            pytest.skip("corpus file not found")
        source = corpus_path.read_text()
        chunks = chunk_code_file("sample_csharp.cs", source, "c_sharp")
        all_text = " ".join(c.text for c in chunks)
        assert "DeliveryMode" in all_text
        assert "EventPriority" in all_text
        assert "RetryPolicy" in all_text
        assert "MaxRetries" in all_text
        micro = [c for c in chunks if len(c.text) < _MIN_CHUNK_CHARS]
        mergeable_micro = []
        for i, c in enumerate(micro):
            ns = c.metadata.get("namespace", "")
            if not ns:
                continue
            prev_ns = chunks[chunks.index(c) - 1].metadata.get("namespace", "") if chunks.index(c) > 0 else ""
            next_idx = chunks.index(c) + 1
            next_ns = chunks[next_idx].metadata.get("namespace", "") if next_idx < len(chunks) else ""
            if ns == prev_ns or ns == next_ns:
                mergeable_micro.append(c)
        assert len(mergeable_micro) == 0, (
            f"Found {len(mergeable_micro)} micro-chunks with same-namespace neighbors: "
            + ", ".join(f"{c.metadata} ({len(c.text)} chars)" for c in mergeable_micro)
        )


class TestGoTypeChunking:
    """Test Go type declarations produce chunks."""

    def test_go_struct_produces_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''package main

type Config struct {
    MaxRequests int
    Window      int
    KeyPrefix   string
}
'''
        chunks = chunk_code_file("config.go", source, "go")
        assert len(chunks) == 1
        assert "Config" in chunks[0].text
        assert "MaxRequests" in chunks[0].text
        assert chunks[0].metadata.get("class") == "Config"

    def test_go_multiple_types_and_funcs(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''package main

type Config struct {
    Max int
}

func NewConfig() Config {
    return Config{Max: 100}
}

type Result struct {
    OK bool
}
'''
        chunks = chunk_code_file("types.go", source, "go")
        class_chunks = [c for c in chunks if c.metadata.get("class")]
        func_chunks = [c for c in chunks if c.metadata.get("function")]
        assert len(func_chunks) == 1
        assert len(class_chunks) >= 1
        all_text = " ".join(c.text for c in chunks)
        assert "Config" in all_text
        assert "Result" in all_text


class TestTopLevelTypeMerge:
    """Test that adjacent top-level type-level chunks merge even with empty namespace."""

    def test_python_enums_merge(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''class Status(Enum):
    ACTIVE = 1
    INACTIVE = 2

class Priority(Enum):
    HIGH = 0
    LOW = 1
'''
        chunks = chunk_code_file("enums.py", source, "python")
        assert len(chunks) == 1
        assert "Status" in chunks[0].text
        assert "Priority" in chunks[0].text

    def test_go_adjacent_structs_merge(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''package main

type Point struct {
    X int
    Y int
}

type Size struct {
    W int
    H int
}
'''
        chunks = chunk_code_file("geo.go", source, "go")
        assert len(chunks) == 1
        assert "Point" in chunks[0].text
        assert "Size" in chunks[0].text

    def test_top_level_functions_dont_merge(self):
        """Top-level functions should NOT merge even when small."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''def a():
    pass

def b():
    pass
'''
        chunks = chunk_code_file("funcs.py", source, "python")
        assert len(chunks) == 2


class TestChunkCodeFileGo:
    """Test Go code chunking."""

    def test_go_functions_produce_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''package main

func Add(a, b int) int {
    return a + b
}

func Subtract(a, b int) int {
    return a - b
}

func Multiply(a, b int) int {
    return a * b
}
'''
        chunks = chunk_code_file("math.go", source, "go")
        assert len(chunks) == 3
        assert all(c.language == "go" for c in chunks)
        assert all(c.content_type == "code" for c in chunks)


class TestChunkCodeFileJava:
    """Test Java code chunking."""

    def test_small_java_class_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''public class Calculator {
    public int add(int a, int b) {
        return a + b;
    }

    public int subtract(int a, int b) {
        return a - b;
    }
}
'''
        chunks = chunk_code_file("Calculator.java", source, "java")
        assert len(chunks) == 1
        assert "Calculator" in chunks[0].text


class TestChunkCodeFileJavaScript:
    """Test JavaScript code chunking."""

    def test_js_functions_produce_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''function greet(name) {
    return "Hello, " + name;
}

function farewell(name) {
    return "Goodbye, " + name;
}

function calculate(x, y) {
    return x + y;
}
'''
        chunks = chunk_code_file("utils.js", source, "javascript")
        assert len(chunks) == 3
        assert all(c.language == "javascript" for c in chunks)


class TestChunkCodeTypeScript:
    """Test TypeScript code chunking."""

    def test_ts_functions_produce_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''function greet(name: string): string {
    return "Hello, " + name;
}

function farewell(name: string): string {
    return "Goodbye, " + name;
}
'''
        chunks = chunk_code_file("utils.ts", source, "typescript")
        assert len(chunks) == 2
        assert all(c.language == "typescript" for c in chunks)


class TestChunkCodeFileBash:
    """Test Bash script chunking."""

    def test_bash_functions_produce_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''#!/bin/bash

greet() {
    echo "Hello, $1"
}

deploy() {
    echo "Deploying..."
    rsync -av src/ dest/
}
'''
        chunks = chunk_code_file("deploy.sh", source, "bash")
        assert len(chunks) == 2
        assert all(c.language == "bash" for c in chunks)

    def test_bash_no_functions_falls_back(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''#!/bin/bash
echo "hello"
ls -la
'''
        chunks = chunk_code_file("simple.sh", source, "bash")
        assert len(chunks) == 1


class TestChunkCodeFileRust:
    """Test Rust code chunking."""

    def test_rust_functions_produce_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn subtract(a: i32, b: i32) -> i32 {
    a - b
}
'''
        chunks = chunk_code_file("math.rs", source, "rust")
        assert len(chunks) == 2
        assert all(c.language == "rust" for c in chunks)


class TestChunkCodeFileRuby:
    """Test Ruby code chunking."""

    def test_ruby_methods_produce_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''def greet(name)
  "Hello, #{name}"
end

def farewell(name)
  "Goodbye, #{name}"
end
'''
        chunks = chunk_code_file("helpers.rb", source, "ruby")
        assert len(chunks) == 2
        assert all(c.language == "ruby" for c in chunks)


class TestChunkCodeFileHTML:
    """Test HTML chunking falls back since HTML has no functions."""

    def test_html_falls_back_to_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''<html>
<head><title>Test</title></head>
<body><h1>Hello</h1></body>
</html>
'''
        chunks = chunk_code_file("index.html", source, "html")
        assert len(chunks) >= 1
        assert all(c.content_type == "code" for c in chunks)


class TestChunkCodeFileC:
    """Test C code chunking."""

    def test_c_functions_produce_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}
'''
        chunks = chunk_code_file("math.c", source, "c")
        assert len(chunks) == 2
        assert all(c.language == "c" for c in chunks)


class TestChunkCodeFileHCL:
    """Test HCL/Terraform code chunking."""

    def test_terraform_blocks_produce_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''
resource "aws_instance" "web" {
  ami           = "ami-123456"
  instance_type = "t2.micro"
}

variable "region" {
  default = "us-east-1"
}

output "ip" {
  value = aws_instance.web.public_ip
}
'''
        chunks = chunk_code_file("main.tf", source, "hcl")
        all_text = " ".join(c.text for c in chunks)
        assert "aws_instance" in all_text
        assert "region" in all_text
        assert "ip" in all_text
        assert all(c.language == "hcl" for c in chunks)

    def test_terraform_block_metadata_has_label(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''
resource "aws_s3_bucket" "data" {
  bucket = "my-bucket"
}
'''
        chunks = chunk_code_file("s3.tf", source, "hcl")
        assert chunks[0].metadata["class"] == "resource.aws_s3_bucket.data"

    def test_tfvars_detected_as_hcl(self):
        assert detect_language("prod.tfvars") == "hcl"


class TestChunkCodeFileDockerfile:
    """Test Dockerfile chunking."""

    def test_dockerfile_detected_by_name(self):
        assert detect_language("Dockerfile") == "dockerfile"
        assert detect_language("Dockerfile.prod") == "dockerfile"
        assert detect_language("path/to/Dockerfile") == "dockerfile"

    def test_dockerfile_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''FROM python:3.13-slim
WORKDIR /app
COPY . .
CMD ["python", "main.py"]
'''
        chunks = chunk_code_file("Dockerfile", source)
        assert len(chunks) == 1
        assert chunks[0].language == "dockerfile"


class TestChunkCodeFileYAML:
    """Test YAML config chunking."""

    def test_yaml_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
spec:
  replicas: 3
'''
        chunks = chunk_code_file("deploy.yaml", source, "yaml")
        assert len(chunks) == 1
        assert chunks[0].language == "yaml"

    def test_yml_extension_detected(self):
        assert detect_language("config.yml") == "yaml"


class TestChunkCodeFileTOML:
    """Test TOML config chunking."""

    def test_toml_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''[tool.poetry]
name = "myapp"
version = "0.1.0"
'''
        chunks = chunk_code_file("pyproject.toml", source, "toml")
        assert len(chunks) == 1
        assert chunks[0].language == "toml"


class TestChunkCodeFileJSON:
    """Test JSON config chunking."""

    def test_json_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '{"name": "test", "version": 1}'
        chunks = chunk_code_file("package.json", source, "json")
        assert len(chunks) == 1
        assert chunks[0].language == "json"

    def test_jsonc_extension_detected(self):
        assert detect_language("tsconfig.jsonc") == "json"


class TestChunkCodeFileXML:
    """Test XML / project file chunking."""

    def test_csproj_detected_and_chunked(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
  </ItemGroup>
</Project>
'''
        chunks = chunk_code_file("MyApp.csproj", source)
        assert len(chunks) >= 1
        assert chunks[0].language == "xml"

    def test_pom_xml_chunked(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''<project>
  <groupId>com.example</groupId>
  <artifactId>myapp</artifactId>
  <version>1.0</version>
</project>
'''
        chunks = chunk_code_file("pom.xml", source)
        assert len(chunks) >= 1
        assert chunks[0].language == "xml"


class TestChunkCodeFileCSS:
    """Test CSS chunking."""

    def test_css_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''body { margin: 0; }
.container { max-width: 1200px; }
'''
        chunks = chunk_code_file("styles.css", source, "css")
        assert len(chunks) >= 1
        assert chunks[0].language == "css"

    def test_scss_detected_as_css(self):
        assert detect_language("theme.scss") == "css"


class TestChunkCodeFileSQL:
    """Test SQL chunking."""

    def test_sql_single_chunk(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = '''CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL
);
'''
        chunks = chunk_code_file("schema.sql", source, "sql")
        assert len(chunks) >= 1
        assert chunks[0].language == "sql"


class TestFilenameDetection:
    """Test extensionless project file detection."""

    def test_gemfile_detected_as_ruby(self):
        assert detect_language("Gemfile") == "ruby"

    def test_rakefile_detected_as_ruby(self):
        assert detect_language("Rakefile") == "ruby"

    def test_pipfile_detected_as_toml(self):
        assert detect_language("Pipfile") == "toml"

    def test_makefile_detected_as_bash(self):
        assert detect_language("Makefile") == "bash"


class TestAllGrammarsLoad:
    """Verify every grammar in the import map loads successfully."""

    @pytest.mark.parametrize("lang", [
        "python", "go", "java", "javascript", "typescript",
        "bash", "html", "c_sharp", "c", "cpp", "rust", "ruby",
        "hcl", "dockerfile", "yaml", "toml", "json",
        "xml", "css", "sql",
    ])
    def test_grammar_loads(self, lang):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        from research_tool.code_chunker import _get_language
        assert _get_language(lang) is not None


class TestFallbackBehavior:
    """Test fallback when tree-sitter can't parse."""

    def test_unsupported_language_falls_back(self):
        source = "some code content here\nmore lines\n"
        chunks = chunk_code_file("data.xyz", source)
        assert len(chunks) == 1
        assert chunks[0].content_type == "code"
        assert "// File: data.xyz" in chunks[0].text
        assert chunks[0].metadata.get("fallback") == "true"

    def test_top_level_statements_no_functions(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = "x = 1\ny = 2\nprint(x + y)\n"
        chunks = chunk_code_file("script.py", source, "python")
        # No functions → fallback to single chunk
        assert len(chunks) == 1
        assert "// File: script.py" in chunks[0].text


class TestContentTypeColumn:
    """Test content_type persistence in store."""

    def test_content_type_stored_and_retrieved(self, tmp_path):
        from research_tool.store import DocumentChunk, ResearchStore, EMBEDDING_DIM
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="file://test.py", title="test.py")
            chunk = DocumentChunk(
                text="def foo(): pass",
                page_url="file://test.py",
                chunk_id="code::chunk-0",
                section_title="foo",
                embedding=[0.1] * EMBEDDING_DIM,
                content_type="code",
            )
            store.store_chunk(chunk)
            retrieved = store.get_chunk("code::chunk-0")
            assert retrieved is not None
            assert retrieved.content_type == "code"
        finally:
            store.close()

    def test_content_type_defaults_to_text(self, tmp_path):
        from research_tool.store import DocumentChunk, ResearchStore, EMBEDDING_DIM
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="http://a.com", title="Test")
            chunk = DocumentChunk(
                text="hello world",
                page_url="http://a.com",
                chunk_id="text::chunk-0",
                section_title="s",
                embedding=[0.1] * EMBEDDING_DIM,
            )
            store.store_chunk(chunk)
            retrieved = store.get_chunk("text::chunk-0")
            assert retrieved is not None
            assert retrieved.content_type == "text"
        finally:
            store.close()

    def test_get_all_chunks_includes_content_type(self, tmp_path):
        from research_tool.store import DocumentChunk, ResearchStore, EMBEDDING_DIM
        store = ResearchStore(db_path=str(tmp_path / "test.db"))
        try:
            store.store_page(url="file://test.py", title="test.py")
            store.store_chunk(DocumentChunk(
                text="code chunk",
                page_url="file://test.py",
                chunk_id="c1",
                section_title="s",
                content_type="code",
            ))
            store.store_chunk(DocumentChunk(
                text="text chunk",
                page_url="file://test.py",
                chunk_id="c2",
                section_title="s",
                content_type="text",
            ))
            chunks = store.get_all_chunks()
            types = {c.chunk_id: c.content_type for c in chunks}
            assert types["c1"] == "code"
            assert types["c2"] == "text"
        finally:
            store.close()


class TestMultiLineCommentPreservation:
    """Test that multi-line Go/Python comments are fully preserved in chunks."""

    GO_SOURCE_WITH_MULTILINE_COMMENTS = '''\
// Package ratelimiter implements a distributed rate limiter using the
// sliding window log algorithm with Redis-backed state.
//
// The sliding window log algorithm tracks individual request timestamps
// within the current window, providing exact rate limiting at the cost
// of O(N) storage per client (where N is the rate limit).
package ratelimiter

import "context"

// ClientKey generates a deterministic key for rate limiting by hashing
// the client identifier with SHA-256. This ensures uniform key distribution
// and prevents key injection attacks from specially crafted identifiers.
func ClientKey(identifier string, prefix string) string {
    return prefix + identifier
}

// FixedWindowCounter implements a simpler fixed-window rate limiting
// algorithm with O(1) storage per client. The window resets at fixed
// intervals, which can allow up to 2x the rate limit at window boundaries
// (the "boundary burst" problem).
type FixedWindowCounter struct {
    config int
}

// Allow checks the fixed-window counter for the given client.
// The window key is computed as Unix timestamp divided by window
// duration in seconds, ensuring all requests within the same
// window period share a counter.
func (f *FixedWindowCounter) Allow(ctx context.Context, clientID string) bool {
    return true
}
'''

    # Build a Python class large enough to exceed _SMALL_CLASS_TOKENS (500 tokens ~ 2000 chars)
    _PYTHON_METHODS = "\n".join(
        f"    def method_{i}(self, value):\n"
        f"        \"\"\"Process value {i} with detailed logic.\"\"\"\n"
        f"        result = value * {i} + {i * 2}\n"
        f"        if result > 100:\n"
        f"            result = result // 2\n"
        f"        return result\n"
        for i in range(15)
    )
    PYTHON_SOURCE_WITH_CLASS_DOCSTRING = (
        'class DependencyGraph:\n'
        '    """Directed acyclic graph for task dependency tracking.\n'
        '\n'
        '    Maintains adjacency lists for both forward dependencies (task A\n'
        '    depends on B) and reverse dependencies (B is depended on by A).\n'
        '    Provides topological ordering and cycle detection.\n'
        '    """\n'
        '\n'
        '    def __init__(self):\n'
        '        self._forward = {}\n'
        '        self._reverse = {}\n'
        '\n'
        + _PYTHON_METHODS
    )

    def test_go_multiline_comment_all_lines_preserved(self):
        """All lines of a multi-line // comment block must appear in the chunk."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("rate.go", self.GO_SOURCE_WITH_MULTILINE_COMMENTS, "go")
        # Find the ClientKey chunk
        client_key_chunks = [c for c in chunks if "ClientKey" in c.text]
        assert len(client_key_chunks) == 1
        text = client_key_chunks[0].text
        # Must contain full doc comment, not just the last line
        assert "deterministic key for rate limiting by hashing" in text
        assert "uniform key distribution" in text
        assert "key injection attacks" in text

    def test_go_full_docstring_not_truncated(self):
        """_enrich_chunk must include full docstring, not just first 100 chars."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("rate.go", self.GO_SOURCE_WITH_MULTILINE_COMMENTS, "go")
        # Find the Allow chunk for FixedWindowCounter
        allow_chunks = [c for c in chunks if "func (f *FixedWindowCounter) Allow" in c.text]
        assert len(allow_chunks) == 1
        text = allow_chunks[0].text
        # Must contain all lines of the preceding comment
        assert "Unix timestamp divided by window" in text
        assert "window period share a counter" in text

    def test_python_class_docstring_carried_to_method_chunks(self):
        """When a large class is split into method chunks, the class docstring must be included."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("dep.py", self.PYTHON_SOURCE_WITH_CLASS_DOCSTRING, "python")
        # DependencyGraph is large enough to be split into method-level chunks
        assert len(chunks) > 1, f"Expected method-level chunks, got {len(chunks)}"
        # Every method chunk should carry the class docstring
        for chunk in chunks:
            assert "adjacency lists for both forward dependencies" in chunk.text, (
                f"Class docstring missing from chunk: {chunk.text[:200]}"
            )

    def test_go_corpus_file_contains_required_passages(self):
        """The actual corpus Go file must produce chunks containing all required passages."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        import pathlib
        corpus_path = pathlib.Path(__file__).parent.parent / "benchmarks" / "corpus" / "sample_go.go"
        if not corpus_path.exists():
            pytest.skip("corpus file not found")
        source = corpus_path.read_text()
        chunks = chunk_code_file("sample_go.go", source, "go")
        all_text = "\n".join(c.text for c in chunks)

        required_passages = [
            "sliding window log algorithm",
            "deterministic key for rate limiting by hashing",
            "can allow up to 2x the rate limit at window boundaries",
            "Unix timestamp divided by window",
            "O(N) storage per client",
        ]
        for passage in required_passages:
            assert passage in all_text, (
                f"Required passage not found in any chunk: {passage!r}"
            )

    def test_python_corpus_file_contains_required_passage(self):
        """The actual corpus Python file must contain the DependencyGraph docstring in method chunks."""
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        import pathlib
        corpus_path = pathlib.Path(__file__).parent.parent / "benchmarks" / "corpus" / "sample_python.py"
        if not corpus_path.exists():
            pytest.skip("corpus file not found")
        source = corpus_path.read_text()
        chunks = chunk_code_file("sample_python.py", source, "python")
        # Find DependencyGraph method chunks (namespace = DependencyGraph)
        dep_graph_chunks = [
            c for c in chunks
            if c.metadata.get("namespace") == "DependencyGraph" and "def " in c.text
        ]
        assert len(dep_graph_chunks) > 0, "No DependencyGraph method chunks found"
        for chunk in dep_graph_chunks:
            assert "adjacency lists for both forward dependencies" in chunk.text, (
                f"Class docstring missing from DependencyGraph method chunk: {chunk.text[:200]}"
            )


class TestModuleDocstringEnrichment:
    """Module-level docstrings should be included in all chunks from the file."""

    PYTHON_WITH_MODULE_DOC = '''\
"""
Event-driven task scheduler with priority queues.

Supports DAG-based execution.
"""

from enum import Enum


class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2


def schedule_task(name):
    return name
'''

    GO_WITH_PACKAGE_COMMENT = '''\
// Package ratelimiter implements a distributed rate limiter.
// Uses sliding window algorithm.
package ratelimiter

func Allow(clientID string) bool {
    return true
}
'''

    CSHARP_WITH_LEADING_COMMENT = '''\
// Notification service for event-driven messaging.
// Supports fire-and-forget and at-least-once delivery.

using System;

public enum DeliveryMode
{
    FireAndForget,
    AtLeastOnce,
}
'''

    def test_python_module_docstring_in_all_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("sched.py", self.PYTHON_WITH_MODULE_DOC, "python")
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "task scheduler with priority queues" in chunk.text

    def test_go_package_comment_in_type_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("rate.go", self.GO_WITH_PACKAGE_COMMENT, "go")
        for chunk in chunks:
            assert "distributed rate limiter" in chunk.text

    def test_csharp_leading_comment_in_chunks(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("notify.cs", self.CSHARP_WITH_LEADING_COMMENT, "c_sharp")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "event-driven messaging" in chunk.text

    def test_no_module_docstring_no_extra_header(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        source = "def foo():\n    return 1\n"
        chunks = chunk_code_file("bare.py", source, "python")
        assert len(chunks) == 1
        lines = chunks[0].text.split("\n")
        assert lines[0] == "// File: bare.py"
        assert lines[1].startswith("def foo")


class TestTypeLevelDocstringEnrichment:
    """Small class chunks should include their own docstring in the header."""

    PYTHON_ENUM_WITH_DOCSTRING = '''\
class Priority(Enum):
    """Task execution priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
'''

    def test_small_class_docstring_in_header(self):
        if not _TREE_SITTER_AVAILABLE:
            pytest.skip("tree-sitter not available")
        chunks = chunk_code_file("pri.py", self.PYTHON_ENUM_WITH_DOCSTRING, "python")
        assert len(chunks) == 1
        header_lines = []
        for line in chunks[0].text.split("\n"):
            if line.startswith("//"):
                header_lines.append(line)
            else:
                break
        header = "\n".join(header_lines)
        assert "Task execution priority levels" in header


class TestBranchMetadata:
    """Branch info should appear in chunk headers when provided."""

    SIMPLE_PYTHON = 'def greet():\n    return "hello"\n'

    @pytest.mark.skipif(not _TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_branch_in_chunk_header(self):
        chunks = chunk_code_file("app.py", self.SIMPLE_PYTHON, "python", branch="5.6")
        assert any("(branch 5.6)" in c.text for c in chunks)

    @pytest.mark.skipif(not _TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_no_branch_no_annotation(self):
        chunks = chunk_code_file("app.py", self.SIMPLE_PYTHON, "python")
        assert not any("(branch" in c.text for c in chunks)

    @pytest.mark.skipif(not _TREE_SITTER_AVAILABLE, reason="tree-sitter not available")
    def test_branch_in_fallback_chunk(self):
        chunks = chunk_code_file("unknown.xyz", "some code", branch="main")
        assert any("(branch main)" in c.text for c in chunks)
