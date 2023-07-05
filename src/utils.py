import collections.abc as collections
SPLITS = {
    'train': [
        'AL', 'BD', 'CD', 'CM', 'GH', 'GU', 'HN', 'IA', 'ID', 'JO', 'KE', 'KM',
        'LB', 'LS', 'MA', 'MB', 'MD', 'MM', 'MW', 'MZ', 'NG', 'NI', 'PE', 'PH',
        'SN', 'TG', 'TJ', 'UG', 'ZM', 'ZW'],
    'val': [
        'BF', 'BJ', 'BO', 'CO', 'DR', 'GA', 'GN', 'GY', 'HT', 'NM', 'SL', 'TD',
        'TZ'],
    'test': [
        'AM', 'AO', 'BU', 'CI', 'EG', 'ET', 'KH', 'KY', 'ML', 'NP', 'PK', 'RW',
        'SZ']
}

COUNTRYS = ['AL', 'AM', 'AO', 'BD', 'BF', 'BJ', 'BO', 'BU', 'CD', 'CI', 'CM',
            'CO', 'DR', 'EG', 'ET', 'GA', 'GH', 'GN', 'GU', 'GY', 'HN', 'HT',
            'IA', 'ID', 'JO', 'KE', 'KH', 'KM', 'KY', 'LB', 'LS', 'MA', 'MB',
            'MD', 'ML', 'MM', 'MW', 'MZ', 'NG', 'NI', 'NM', 'NP', 'PE', 'PH',
            'PK', 'RW', 'SL', 'SN', 'SZ', 'TD', 'TG', 'TJ', 'TZ', 'UG', 'ZM',
            'ZW']

FOLDS = {
    0: ['MB', 'GY', 'KM', 'SZ', 'NI', 'KY', 'GA', 'AM', 'MM', 'MA', 'CI', 'AL', 'TD', 'TJ', 'TG', 'CD', 'GU', 'BO', 'AO', 'LS'],
    1: ['HN', 'BU', 'MZ', 'NP', 'NM', 'ID', 'GN', 'ZW', 'LB'],
    2: ['PK', 'CM', 'HT', 'ZM', 'BJ', 'BF', 'DR'],
    3: ['SL', 'RW', 'GH', 'MD', 'CO', 'ML'],
    4: ['KE', 'KH', 'UG', 'PH', 'SN'],
    5: ['ET', 'TZ', 'BD', 'JO'],
    6: ['MW', 'NG', 'PE'],
    7: ['EG', 'IA']
}

RARE_FOLDS = {
    0: ['MB', 'GY', 'KM', 'SZ', 'NI', 'KY', 'GA', 'AM', 'MM', 'MA', 'CI', 'AL', 'TD', 'TJ', 'TG', 'CD', 'GU', 'BO', 'AO'],
    1: ['HN', 'BU', 'MZ', 'NP', 'NM', 'ID', 'GN', 'ZW', 'LB'],
    2: ['PK', 'CM', 'ZM', 'BJ', 'BF', 'DR'],
    3: ['SL', 'RW', 'GH', 'MD', 'CO', 'ML'],
    4: ['KE', 'KH', 'UG', 'PH', 'SN'],
    5: ['ET', 'TZ', 'BD', 'JO', 'IA', 'MW', 'NG'],
    6: ['PE'],
    7: ['EG'],
    8: ['LS'],
    9: ['HT']
}


AFRICA = {
    0: ['AO', 'CI', 'ET', 'ML', 'RW'],
    1: ['BJ', 'BF', 'GN', 'SL', 'TZ'],
    2: ['CM', 'GH', 'MW', 'ZW'],
    3: ['CD', 'MZ', 'NG', 'TG', 'UG'],
    4: ['KE', 'LS', 'SN', 'ZM'],
}

AFRICA_2 = {
    0: ['AO', 'CI', 'ET', 'ML', 'RW'],
    1: ['BJ', 'BF', 'GN', 'SL', 'TZ'],
    2: ['CM', 'GH', 'MW', 'ZW'],
    3: ['CD', 'MZ', 'NG', 'TG', 'UG'],
    4: ['BU', 'EG', 'GA', 'KM', 'LB', 'MD', 'MA', 'NI', 'RW', 'SZ', 'NM'],
    5: ['KE'],
    6: ['LS'],
    7: ['SN'],
    8: ['ZM'],
}


def get_patch(sample, path='data'):
    folder = str(sample)[:10]
    image = str(sample)
    return path + '/images/' + folder + '/' + image + '.npz'


def deep_update(source, overrides):
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source
