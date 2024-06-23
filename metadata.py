from pathlib import Path
import os

import pandas as pd
import pytz
from win32com.propsys import propsys, pscon

TRAIN_PATH = Path(r'D:\Traffic_Counting_Videos')
TEST_PATH = Path(r'D:\Traffic_Counting_Test')

items = []

train_lookup = {
    (1, 4): (53.2354515772493, 10.397761072740538),
    (5, 9): (53.264802202381105, 10.406664706791815),
    (10, 15): (53.26164571178299, 10.421726222400766),
    (16, 22): (53.23875944105805, 10.41311675728099),
    (23, 27): (53.26361835887896, 10.407229843809962),
    (28, 32): (53.24527563853708, 10.41609687864867),
    (33, 38): (53.22878987445729, 10.405535405680919),
    (39, 40): (53.229014929809324, 10.405387755589343),
    (41, 43): (53.2612368957192, 10.421094088663052),
    (44, 46): (53.250163589737866, 10.398332381287439)
}

test_lookup = {
    'low': (53.23162048908169, 10.398868492540936),
    'moderate': (53.238723612475205, 10.399540368335424),
    'high': (53.238765373779344, 10.413225059825775)
}

for file in os.listdir(TRAIN_PATH):
    file_path = TRAIN_PATH / file
    filename = file.split('.')[0]
    number = int(filename.split('_')[1])

    # Get the creation time
    properties = propsys.SHGetPropertyStoreFromParsingName(file_path.__str__())
    dt = properties.GetValue(pscon.PKEY_Media_DateEncoded).GetValue()
    converted_time = dt.astimezone(pytz.timezone('Europe/Malta'))

    # Get the location
    lat, lon = None, None
    for (lower, upper), value in train_lookup.items():
        if lower <= number <= upper:
            lat, lon = value
            break

    items.append((['train', filename, converted_time, lat, lon]))

for file in os.listdir(TEST_PATH):
    file_path = TEST_PATH / file
    filename = file.split('.')[0]

    # Get the creation time
    properties = propsys.SHGetPropertyStoreFromParsingName(file_path.__str__())
    dt = properties.GetValue(pscon.PKEY_Media_DateEncoded).GetValue()
    converted_time = dt.astimezone(pytz.timezone('Europe/Malta'))

    # Get the location
    lat, lon = test_lookup[filename.split('_')[0].lower()]

    items.append((['test', filename, converted_time, lat, lon]))

metadata = pd.DataFrame(data=items, columns=['subset', 'filename', 'timestamp', 'lat', 'lon'])
