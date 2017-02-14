#include "cy_handler.h"
#include "Python.h"
#include <exception>
#include <string>
#include "HalconCpp.h"


void raise_py_error()
{
  try {
    throw;
  } 
  catch (HalconCpp::HException& e) {
    HalconCpp::HString msg = e.ErrorMessage();
    PyErr_SetString(PyExc_RuntimeError, msg.Text());
  }
}
