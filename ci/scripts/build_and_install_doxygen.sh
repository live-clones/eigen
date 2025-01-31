git clone --depth 1 --branch $1 https://github.com/doxygen/doxygen.git
cmake -B .build -G Ninja
cmake --build .build
mv .build/bin/doxygen /usr/local/bin
