from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMake, cmake_layout, CMakeDeps
from conan.tools.system.package_manager import Apt

class bcpdRecipe(ConanFile):
    name = "bcpd"
    version = "1.0"

    # Optional metadata
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of hello package here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")

    # Binary configuration
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    # Sources are located in the same place as this recipe, copy them to the recipe
    exports_sources = "CMakeLists.txt", "src/*", "include/*"

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)
        #self.source_folder = "src"
        #self.folders.build = "build"
        #self.folders.generators = "CMakeFiles"

    def requirements(self):
        self.requires("catch2/3.4.0")
        self.requires("eigen/3.4.0")
        self.requires("cvplot/1.2.2")
        self.requires("xz_utils/5.4.4", override=True)

    def system_requirements(self):
        apt = Apt(self)
        #apt.install(["libgtk2.0-dev"] if self.options.version == 2 else ["libgtk-3-dev"], update=True, check=True)
        apt.install(["libva-dev"], update=True, check=True)

    def generate(self):
        cmake = CMakeDeps(self)
        cmake.generate()

        tc = CMakeToolchain(self)
        tc.user_presets_path = "ConanPresets.json"
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["bcpd"]
