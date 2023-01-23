# CMake Guide

This is a short guide on how to incorporate Eigen into to your own project with the help of CMake. If you are not familiar with Eigen at all, please take a look at the [getting started guide](https://eigen.tuxfamily.org/dox/GettingStarted.html) first.

## Prerequisites

1. CMake:
    - Version 3.0 or higher.
    - Run ```cmake --version``` to check which version is installed.
2. Eigen:
    - Since Eigen is a pure template library, you only need to make sure that the header files are available on your system.
    - One way to do this, is to simply clone the repository:
        ```bash
        git clone https://gitlab.com/libeigen/eigen.git
        ```

## A Minimal Example

The following is an easy and minimal example to get you started with CMake and Eigen. An explanation of the CMake commands used in this example can be found in the section afterwards. 

<!-- Note that this is not neccessarily the best way to use Eigen with CMake. However, this is a good starting point for people that are new to CMake and Eigen. A slightly advanced example that makes use of Eigen's native CMake support can be found in the *An Advanced Example* section. -->

In order to follow this example, create a folder *myproject* (you can actually name it however you want) and place the eigen header files there.

The structure should look like this:

```tree
myproject/
    `-- eigen/
```

We will now create a simple program, which will make use of the Eigen library. Create a file named *example.cpp* with the following contents:

```C++
 1 #include <iostream>
 2 #include <Eigen/Dense>
 3
 4 using Eigen::MatrixXd;
 5 
 6 int main()
 7 {
 8   MatrixXd m(2,2);
 9   m(0,0) = 3;
10   m(1,0) = 2.5;
11   m(0,1) = -1;
12   m(1,1) = m(1,0) + m(0,1);
13   std::cout << m << std::endl;
14 }
```

If you completed the [getting started guide](https://eigen.tuxfamily.org/dox/GettingStarted.html) you should already be familiar with this example.

Now we need to tell CMake what do to. Therefore, we create a file named *CMakeLists.txt* with the following contents: 

```cmake
1 cmake_minimum_required (VERSION 3.0)
2 project (myproject)
4 include_directories("eigen")
5 add_executable (example example.cpp)
```

The directory structure should now look like this:
```tree
myproject/
    |-- CMakeLists.txt
    |-- example.cpp
    `-- eigen/
```

### Running CMake

Now we are technically ready to run CMake. However, running CMake will create various build files that can clutter the directory. Therefore it is recommended to create a build folder and run CMake within that folder.

```bash
mkdir build
cd build
cmake ..
```

Or as a one-liner
```bash
mkdir build && cd build && cmake..
```

The directory structure should now be the following
```tree
myproject/
    |-- CMakeLists.txt
    |-- example.cpp
    `-- eigen/
    `__ build/
        `-- cmake_install.cmake
        `-- CMakeCache.txt
        `-- Makefile
        `-- CMakeFiles/
```

Now, from within the build directory, you can use the *make* command to create the executable.

```bash
make
```

This will add one more file to the build folder, which is the executable of our example program. Execute your program with

```bash
./example
```

You should now see the output

```bash
  3  -1
2.5 1.5
```

### Explanation of the CMake File

This section provides an explanation of the commands that we used in our small CMakeLists.txt. These commands are typically used in every CMakeLists.txt.

1. cmake_minimum_required (VERSION 3.0)
    - The top most command in any *CMakeLists.txt* must specify the minimum required CMake version.
    - In Eigen's case the minimum required version is 3.0.

2. project (myproject)
    - This command sets the name of the project and is also required.
    - Typically it should be used right after *cmake_minimum_required*.

3. include_directories ("eigen")
    - This command adds the specified directory to those that the compiler uses to search for include files.
    - This has to point to the eigen header files, otherwise the compiler will not be able to find them.
    - In our example, the eigen header files lie within the project directory in the eigen folder (.../myproject/eigen). If you placed them outside, make sure that the path to the eigen directory is correctly set in this command.

4. add_executable (example example.cpp)
    - This command tells CMake to create an executable (named *example*) using the specified source code files.

<!-- ## An Advanced Example

Eigen provides native CMake support, by exporting a CMake target called Eigen3::Eigen. This CMake target can be imported via the *find_package* CMake command and used by calling *target_link_libraries*. Th -->

