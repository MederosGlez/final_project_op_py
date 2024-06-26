
from tools import _handle_gen, load_database, save_database
from main import process_line
import os


def test_0_save_database():
    process_line("f(x) = x + 1")
    save_database()
    assert os.path.exists('.save_functions.txt') == True


def test_1_load_database():
    load_database()
    assert process_line('f(2)') == 3


def test_2_handle_gen():
    load_database()
    assert _handle_gen(('f', 1)) == 2
