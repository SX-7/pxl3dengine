import pytest
import logging
from utils import Vec4, Shape, _verify_type

logger = logging.getLogger("__Vec4__")
logger.setLevel(logging.INFO)

def test_error_raising():
    with pytest.raises(TypeError):
        _verify_type(Vec4(1,2,3,4),bool,str,float)
    with pytest.raises(TypeError):
        _verify_type(1,bool,str,float)    
    with pytest.raises(TypeError):
        _verify_type(Shape,bool,str,float)

def test_type_index():
    index = _verify_type(Vec4(1,2,3,4),bool,Shape,Vec4,float,int,str)
    assert index == 2
    index = _verify_type(Shape([]),bool,Shape,Vec4,float,int,str)
    assert index == 1
    index = _verify_type("15",bool,Shape,Vec4,float,int,str)
    assert index == 5