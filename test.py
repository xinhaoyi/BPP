import re

if __name__ == '__main__':
    pathway = re.sub(r"\s+", '_', "Cell-Cell communication")
    print(pathway)
