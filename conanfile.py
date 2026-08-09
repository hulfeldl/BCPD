from conan import ConanFile
from conan.tools.cmake import cmake_layout

class BCPDConan(ConanFile):
    name = "BCPD"
    version = "0.1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"
    requires = ["fmt/11.0.2", "spdlog/1.14.1"]
    def layout(self):
        cmake_layout(self)
