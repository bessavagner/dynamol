import os


def mkdir(dirname, silent=False):
    if not os.path.exists(dirname):
        try:
            os.mkdir(dirname)
            print(f"Pasta '{dirname}' criada.")
        except FileNotFoundError as err:
            print("Erro: ", err)
    elif not silent:
        print(f"'{dirname}' já existe.")

    return dirname
