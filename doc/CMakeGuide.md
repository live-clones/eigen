# CMake Guide

This is a short guide on how to incorporate Eigen into to your own project with the help of CMake. If you are not familiar with Eigen at all, please take a look at the [getting started guide](https://eigen.tuxfamily.org/dox/GettingStarted.html) first.

## Prerequisites

1. CMake:
    - Version 3.0 or higher.
    - Run ```cmake --version``` to check which version is installed.
2. Eigen:
    - Eigen is a pure template library, therefore, you only need to make sure that the header files are available on your system.
    - One way to do this, is to simply clone the repository:
        ```bash
        git clone https://gitlab.com/libeigen/eigen.git
        ```

Note that there are multiple ways of using Eigen with CMake. This guide provides two of them. The first one is very straightforward and is targeted for people that are new to CMake and Eigen. The second one makes full use of Eigen's native CMake support and is therefore the recommended way. However, it is ever so slightly more difficult and requires the installation of Eigen. Both guides are written independently, if you want to follow the recommended way from the get go, feel free to do so. An explanation of all CMake commands that are used in this guide can be found at the end. 

## The *Simple* Way

The following is an easy and straightforward example to get you started with CMake and Eigen.

### Preparing the Files

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

## Using Eigen's Native CMake Support

Eigen provides native CMake support, by exporting a CMake target called Eigen3::Eigen. This CMake target can be imported via the *find_package* CMake command and used by calling *target_link_libraries*.

### Installing Eigen

In order to follow this example, you first have to make sure that Eigen is installed.

Move to the directory where the Eigen source files are located. This directory should include a file called *INSTALL*, which provides instructions on how to install Eigen using CMake. Feel free to take a look at it, and if not, just execute the following commands. 

Assuming you are in the Eigen source directory
```bash
mkdir build
cd build
cmake ..
make install
```

The *make install* command may require administrator privileges. In that the case, simply do 
```bash
sudo make install
```

### Preparing the Files

Now that Eigen is installed, create a folder *myproject* (wherever you want). 

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
3 find_package (Eigen3 3.3 REQUIRED NO_MODULE)
4 add_executable (example example.cpp)
5 target_link_libraries (example Eigen3::Eigen)
```

The directory structure should now look like this:
```tree
myproject/
    |-- CMakeLists.txt
    |-- example.cpp
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

## Explanation of the CMake Commands

The three following commands are used in both methods and are typically part of every CMakeLists.txt.

**cmake_minimum_required** (VERSION 3.0)
  - The top most command in any *CMakeLists.txt* must specify the minimum required CMake version.
  - In Eigen's case the minimum required version is 3.0.

**project** (myproject)
  - This command sets the name of the project and is also required.
  - Typically it should be used right after *cmake_minimum_required*.

**add_executable** (example example.cpp)
  - This command tells CMake to create an executable (named *example*) using the specified source code files.

We used this command only in the first example.

**include_directories** ("eigen")
  - This command adds the specified directory to those that the compiler uses to search for include files.
  - This has to point to the eigen header files, otherwise the compiler will not be able to find them.
  - In our example, the eigen header files lie within the project directory in the eigen folder (.../myproject/eigen). If you placed them outside, make sure that the path to the eigen directory is correctly set in this command.

We used these commands only in the second example.

**find_package** (Eigen3 3.3 REQUIRED NO_MODULE)
  - This command is used to find *cmake-ready* external package on the system and make it available.
  - In our example we are looking for the Eigen3 package in version 3.3 or higher.

**target_link_libraries** (example Eigen3::Eigen)
  - Links the library into the target executable.
  - In this case the Eigen3::Eigen target is linked to the example program.
