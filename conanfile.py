from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout

class BCPDConan(ConanFile):
    name = "bcpd"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def requirements(self):
        # Add BCPD-specific dependencies here
        # self.requires("fmt/10.1.1")
        # self.requires("spdlog/1.13.0")
        pass

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()

    def layout(self):
        cmake_layout(self)
