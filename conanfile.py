from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, CMake, cmake_layout

class BCPDConan(ConanFile):
    name = "bcpd"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    
    def requirements(self):
        self.requires("catch2/3.5.0")
        self.requires("spdlog/1.14.1")
        self.requires("nanoflann/1.9.0")
        self.requires("tinyply/2.3.4")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.generate()
        deps = CMakeDeps(self)
        deps.generate()
    
    def layout(self):
        cmake_layout(self)
