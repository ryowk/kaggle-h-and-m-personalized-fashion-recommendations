ARTICLES_ORIGINAL = {
    'article_id': 'object',
    'product_code': 'int64',
    'prod_name': 'object',
    'product_type_no': 'int64',
    'product_type_name': 'object',
    'product_group_name': 'object',
    'graphical_appearance_no': 'int64',
    'graphical_appearance_name': 'object',
    'colour_group_code': 'int64',
    'colour_group_name': 'object',
    'perceived_colour_value_id': 'int64',
    'perceived_colour_value_name': 'object',
    'perceived_colour_master_id': 'int64',
    'perceived_colour_master_name': 'object',
    'department_no': 'int64',
    'department_name': 'object',
    'index_code': 'object',
    'index_name': 'object',
    'index_group_no': 'int64',
    'index_group_name': 'object',
    'section_no': 'int64',
    'section_name': 'object',
    'garment_group_no': 'int64',
    'garment_group_name': 'object',
    'detail_desc': 'object',
}

ITEMS = [
    'item',
    'product_type_no_idx',
    'product_group_name_idx',
    'graphical_appearance_no_idx',
    'colour_group_code_idx',
    'perceived_colour_value_id_idx',
    'perceived_colour_master_id_idx',
    'department_no_idx',
    'index_code_idx',
    'index_group_no_idx',
    'section_no_idx',
    'garment_group_no_idx',
]

CUSTOMERS_ORIGINAL = {
    'customer_id': 'object',
    'FN': 'float64',
    'Active': 'float64',
    'club_member_status': 'object',
    'fashion_news_frequency': 'object',
    'age': 'float64',
    'postal_code': 'object',
}

USERS = [
    'user',
    'FN',
    'Active',
    'club_member_status_idx',
    'fashion_news_frequency_idx',
    'age',
]

TRANSACTIONS_ORIGINAL = {
    'customer_id': 'object',
    'article_id': 'object',
    'price': 'float64',
    'sales_channel_id': 'int64',
}

TRANSACTIONS = [
    't_dat',
    'user',
    'item',
    'price',
    'sales_channel_id',
]

SAMPLE_SUBMISSION_ORIGINAL = {
    'customer_id': 'object',
    'prediction': 'object',
}

SAMPLE_SUBMISSION = [
    'customer_id',
    'user',
]
