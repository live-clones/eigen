# CMake Guide

This is a short guide on how to incorporate Eigen into to your own project with the help of CMake. If you are not familiar with Eigen at all, please take a look at the [getting started guide](https://eigen.tuxfamily.org/dox/GettingStarted.html) first.

## Prerequisites

Before we start, you have to make sure that both CMake and Eigen are available on your system. 

**CMake:** The recommended CMake version is 3.10 or higher. Run ```cmake --version``` to check which version is installed on your system. If the version is below 3.10 or CMake is not installed at all, make sure to update/install it before you continue. Note that not all parts of this guide require CMake version 3.10. However, having CMake version 3.10 or higher ensures that everything in this guide can be followed.

**Eigen:** There are multiple ways to make Eigen available on your system. Since Eigen is a pure template library, it is sufficient to download the header files to a local folder and access them from there. However, you can also install Eigen, which can be more convenient.

### Obtaining the Source Files

You can obtain the source files by cloning the official repository:
```bash
git clone https://gitlab.com/libeigen/eigen.git
```

Another option is to simply download and extract one of the provided archives from the [main page](https://eigen.tuxfamily.org/index.php?title=Main_Page):
```bash
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvfz eigen-3.4.0.tar.gz
```

### Installing Eigen

Eigen can also be installed to make it available localy or system-wide. 

If you are on a linux system, your package manager might provide the Eigen package. In this case you don't need to download the source files. 

On debian based systems simply run:
```bash
sudo apt install libeigen3-dev
```

If Eigen is not provided by your package manager or you want to install it manually, you first need to obtain the source files as described in *Obtaining the Source Files*. 

Move to the directory where the Eigen source files are located. This directory should include a file called *INSTALL*, which provides instructions on how to install Eigen using CMake.

Assuming you are in the Eigen source directory:
```bash
cmake -B build
cmake --build build
```

The ```install``` target may require administrator privileges. In that the case, simply do 
```bash
cmake --install build
```

You can also install to a local directory without administrator privileges:
```bash
cmake -B build -DCMAKE_INSTALL_PREFIX=<path to local installation folder>
cmake --build build
cmake --install build
```

## Preparing the Files

Now that Eigen is available on your system, create a folder *myproject* (wherever you want). 

We will now create a simple program, which will make use of the Eigen library. Create a file named *example.cpp* with the following contents:

```C++
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;

int main()
{
  MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
}
```

If you completed the [getting started guide](https://eigen.tuxfamily.org/dox/GettingStarted.html) you should already be familiar with this example.

Now we need to tell CMake what do to. Therefore, we create a file named *CMakeLists.txt*. The contents depend on whether you use the Eigen source files from a specific local folder or through the installation.

For Eigen in a local folder:

```cmake
cmake_minimum_required (VERSION 3.10)
project (myproject)
target_include_directories("<path to eigen source files>")
add_executable (example example.cpp)
```

For the installed version:
```cmake
cmake_minimum_required (VERSION 3.10)
project (myproject)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
add_executable (example example.cpp)
target_link_libraries (example Eigen3::Eigen)
```

The directory structure should now look like this:
```tree
myproject/
    |-- CMakeLists.txt
    |-- example.cpp
```

## Running CMake

Now we are technically ready to run CMake. However, running CMake will create various build files that can clutter the directory. Therefore it is recommended to create a build folder and run CMake within that folder. The following command does all of this.

```bash
cmake -B build
```

---
**Note**: If Eigen is not installed in a default location or if you want to pick a specific version, you have to set the *CMAKE_PREFIX_PATH* variable:
```bash
cmake -B build -DCMAKE_PREFIX_PATH=$HOME/mypackages
```

Alternatively, you can also set the *Eigen3_DIR* variable to the respective path containing the Eigen3*.cmake files:
```bash
cmake -B build -DEigen3_DIR=$HOME/mypackages/share/eigen3/cmake/
```
---
The directory structure should now be the following
```tree
myproject/
    |-- CMakeLists.txt
    |-- example.cpp
    `__ build/
        `-- cmake_install.cmake
        `-- CMakeCache.txt
        `-- Makefile
        `-- CMakeFiles/
```

Now you can use the following command to create the executable.

```bash
cmake --build build
```

This will add one more file to the build folder, which is the executable of our example program. Execute your program with

```bash
./example (from within the build folder)
```

You should now see the output

```bash
  3  -1
2.5 1.5
```

## Explanation of the CMake Commands

The three following commands are used in both cases and are typically part of every CMakeLists.txt.

[**cmake_minimum_required**](https://cmake.org/cmake/help/v3.5/command/cmake_minimum_required.html) (VERSION 3.0)
  - The top most command in any *CMakeLists.txt* must specify the minimum required CMake version.
  - In Eigen's case the minimum required version is 3.0.

[**project**](https://cmake.org/cmake/help/latest/command/project.html) (myproject)
  - This command sets the name of the project and is also required.
  - Typically it should be used right after *cmake_minimum_required*.

[**add_executable**](https://cmake.org/cmake/help/latest/command/add_executable.html) (example example.cpp)
  - This command tells CMake to create an executable (named *example*) using the specified source code files.

The following command is neccessary if you access Eigen from a local folder:

[**target_include_directories**](https://cmake.org/cmake/help/latest/command/include_directories.html) ("\<path to eigen source files\>")
  - This command adds the specified directory to those that the compiler uses to search for include files.
  - This has to point to the eigen header files, otherwise the compiler will not be able to find them.
  - In our example, the eigen header files lie within the project directory in the eigen folder (.../myproject/eigen). If you placed them outside, make sure that the path to the eigen directory is correctly set in this command.

These commands are neccessary if you use an installed version of Eigen. This makes use Eigen's native CMake support, by importing the CMake target Eigen3::Eigen:

[**find_package**](https://cmake.org/cmake/help/latest/command/find_package.html) (Eigen3 3.3 REQUIRED NO_MODULE)
  - This command is used to find *cmake-ready* external package on the system and make it available.
  - In our example we are looking for the Eigen3 package in version 3.3 or higher.

[**target_link_libraries**](https://cmake.org/cmake/help/latest/command/target_link_libraries.html) (example Eigen3::Eigen)
  - Links the library into the target executable.
  - In this case the Eigen3::Eigen target is linked to the example program.
