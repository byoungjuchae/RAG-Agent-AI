#include <Python.h>
#include <string>

// C++ 함수
std::string greet(const std::string& name) {
    return "Hello, " + name + "!";
}

// Python에서 호출 가능한 래퍼 함수
static PyObject* py_greet(PyObject* self, PyObject* args) {
    const char* name;
    if (!PyArg_ParseTuple(args, "s", &name)) {
        return NULL;
    }
    std::string result = greet(name);
    return Py_BuildValue("s", result.c_str());
}

// 함수 목록
static PyMethodDef ExampleMethods[] = {
    {"greet", py_greet, METH_VARARGS, "Greet the user"},
    {NULL, NULL, 0, NULL} // 종료 표시
};

// 모듈 정의
static struct PyModuleDef examplemodule = {
    PyModuleDef_HEAD_INIT,
    "example",
    "Example Module using Python C API",
    -1,
    ExampleMethods
};

// 모듈 초기화
PyMODINIT_FUNC PyInit_example(void) {
    return PyModule_Create(&examplemodule);
}
