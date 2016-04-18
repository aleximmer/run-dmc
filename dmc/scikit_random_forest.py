from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt
import csv
import time
import datetime

ORDER_CLASS_PATH = 'task/orders_class.txt'
ORDER_TRAIN_PATH = 'task/orders_train.txt'

ORDER_TRAIN_10_000_PATH = 'sample_data/orders_train_10_000.csv'
ORDER_TRAIN_20_000_PATH = 'sample_data/orders_train_20_000.csv'
ORDER_TRAIN_100_000_PATH = 'sample_data/orders_train_100_000.csv'

PREPROCESSED_PATH = 'sample_data/orders_train_preprocessed.csv'
PREPROCESSED_TEST_PATH = 'sample_data/preprocessed_test_data.csv'

PAYMENT_METHODS = set(['RG', 'CBA', 'NN', 'KKE', 'BPRG', 'PAYPALVC', 'VORAUS', 'BPLS', 'BPPL', 'KGRG'])
NA_VALUE = -1

def preprocess_row(row):
    # orderID, orderDate, articleID, colorCode, sizeCode, productGroup, quantity, price, rrp, voucherID, voucherAmount, customerID, 
    # deviceID, is_CBA, is_NN, is_PAYPALVC, is_RG, is_BPRG, is_KKE, is_VORAUS, is_BPLS, is_BPPL, is_KGRG, returnQuantity

    # Clean orderID
    row[0] = row[0].split('a')[1]

    # Convert orderDate to timestamp
    row[1] = time.mktime(datetime.datetime.strptime(row[1], "%Y-%m-%d").timetuple())

    # Clean articleID
    row[2] = row[2].split('i')[1]

    # Clean sizeCode
    size_code_dict = {"A": 0, "I": 1, "L": 2, "M": 3, "S": 4}
    if row[4] in size_code_dict:
        row[4] = size_code_dict[row[4]]

    # Clean productGroup
    if "NA" in row[5]:
        row[5] = NA_VALUE

    # Clean rrp
    if "NA" in row[8]:
        row[8] = NA_VALUE

    # Clean voucherID
    if 'v' in row[9]:
        row[9] = row[9].split('v')[1]
    if "NA" in row[9]:
        row[9] = NA_VALUE

    # Clean customerID
    row[11] = row[11].split('c')[1]

    # paymentMethod -> is_RG, is_CBA, ...
    # "PAYPAL" -> is_RG(0), ..., is_PAYPAL(1)
    unprocessed_payment_method = row[13]
    one_hot_payment_methods = [ "1" if unprocessed_payment_method == payment_method else "0" for payment_method in PAYMENT_METHODS]
    row = row[:13] + list(one_hot_payment_methods) + list(row[-1])

    return row


def get_preprocessed_header(unprocessed_headers):
    # paymentMethod -> is_RG, is_CBA, ...
    # orderID, orderDate, articleID, colorCode, sizeCode, productGroup, quantity, price, rrp, voucherID, voucherAmount, customerID, 
    # deviceID, is_CBA, is_NN, is_PAYPALVC, is_RG, is_BPRG, is_KKE, is_VORAUS, is_BPLS, is_BPPL, is_KGRG, returnQuantity
    processed_headers = unprocessed_headers[:13]
    processed_headers.extend(['is_' + i for i in PAYMENT_METHODS])
    processed_headers.append(unprocessed_headers[-1])
    return processed_headers

# Take the dataset and save the preprocessed dataset in an extra CSV file
def save_preprocessed_data(training_dataset_path, preprocessed_dataset_path):
    # Read unprocessed data
    with open(training_dataset_path, "rb") as unprocessed_file:
        unprocessed_reader = csv.reader(unprocessed_file, delimiter=';')
        unprocessed_rows = []
        for row in unprocessed_reader:
            unprocessed_rows.append(row)

    # Process data
    preprocessed_rows = list()
    unprocessed_header = unprocessed_rows[0]
    preprocessed_rows.append(get_preprocessed_header(unprocessed_header))
    for row in unprocessed_rows[1:]:
        preprocessed_rows.append(preprocess_row(row))

    # Write processed data   
    with open(preprocessed_dataset_path,"wb") as processed_file:
        processed_writer = csv.writer(processed_file, delimiter=';', lineterminator='\r\n', quoting = csv.QUOTE_NONE)
        for row in preprocessed_rows:
            processed_writer.writerow(row)

def main(preprocessed_dataset_path):
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open(preprocessed_dataset_path,'r'), delimiter=';', dtype='f8')[1:]
    # target label is in last column    
    target = [x[-1] for x in dataset]
    train = [x[:-1] for x in dataset]
    
    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)

    # Test Classifier
    test_data_path = ORDER_TRAIN_100_000_PATH
    preprocessed_test_data_path = PREPROCESSED_TEST_PATH
    save_preprocessed_data(test_data_path, preprocessed_test_data_path)
    test_data = genfromtxt(open(preprocessed_test_data_path,'r'), delimiter=';', dtype='f8')[1:]
    test_target = [x[-1] for x in test_data]
    test_train = [x[:-1] for x in test_data]
    print rf.score(test_train, test_target)

if __name__ == "__main__":
    training_dataset_path = ORDER_TRAIN_20_000_PATH
    preprocessed_dataset_path = PREPROCESSED_PATH
    save_preprocessed_data(training_dataset_path, preprocessed_dataset_path)
    main(preprocessed_dataset_path)
