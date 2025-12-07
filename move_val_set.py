import os
DATA_ROOT = "ACDC/database/"
VALIDATION_RANGE = range(86,101)


def main():
    os.makedirs(os.path.join(DATA_ROOT, 'validation'), exist_ok=True)
    for i in VALIDATION_RANGE:
        folder_name = f'patient{i:03d}'
        os.rename(os.path.join(DATA_ROOT, "training", folder_name),
                  os.path.join(DATA_ROOT, 'validation', folder_name))
        
if __name__ == '__main__':
    main()