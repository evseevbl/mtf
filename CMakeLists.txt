cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
project(mtf)

find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    list(APPEND ThirdParty_LIBS ${OpenCV_LIBS})
    include_directories(${OpenCV_INCLUDE_DIRS})
endif (OpenCV_FOUND)

add_executable(${PROJECT_NAME} main.cpp mtf.cpp mtf.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES ENABLE_EXPORTS on)
