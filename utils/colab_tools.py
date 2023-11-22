from IPython.display import Code
# Функция для показа кода из файла в колабе
def show_file_as_code(filename):
    with open(filename, "r") as fp:
       display(Code("".join(fp.readlines())))
       