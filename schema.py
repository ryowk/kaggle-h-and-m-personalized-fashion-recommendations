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

ARTICLES = {
    'article_id_idx': 'int64',
    'product_type_no_idx': 'int64',
    'product_group_name_idx': 'int64',
    'graphical_appearance_no_idx': 'int64',
    'colour_group_code_idx': 'int64',
    'perceived_colour_value_id_idx': 'int64',
    'perceived_colour_master_id_idx': 'int64',
    'department_no_idx': 'int64',
    'index_code_idx': 'int64',
    'index_group_no_idx': 'int64',
    'section_no_idx': 'int64',
    'garment_group_no_idx': 'int64',
}

CUSTOMERS_ORIGINAL = {
    'customer_id': 'object',
    'FN': 'float64',
    'Active': 'float64',
    'club_member_status': 'object',
    'fashion_news_frequency': 'object',
    'age': 'float64',
    'postal_code': 'object',
}

CUSTOMERS = {
    'customer_id_idx': 'int64',
    'FN': 'int64',
    'Active': 'int64',
    'club_member_status_idx': 'int64',
    'fashion_news_frequency_idx': 'int64',
    'age': 'float64',
}

TRANSACTIONS_ORIGINAL = {
    'customer_id': 'object',
    'article_id': 'object',
    'price': 'float64',
    'sales_channel_id': 'int64',
}

TRANSACTIONS = {
    'customer_id_idx': 'int64',
    'article_id_idx': 'int64',
    'price': 'float64',
    'sales_channel_id': 'int64',
}

SAMPLE_SUBMISSION_ORIGINAL = {
    'customer_id': 'object',
    'prediction': 'object',
}

SAMPLE_SUBMISSION = {
    'customer_id': 'object',
    'customer_id_idx': 'int64',
}
