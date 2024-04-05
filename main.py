from src import process_data
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    process_data.preprocess_data()
    process_data.build_loaders()
    print('loaders_printed_successfully')


