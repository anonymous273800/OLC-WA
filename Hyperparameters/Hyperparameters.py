
# OLC_WA
olc_wa_batch_size = 10
olc_wa_w_inc = .5  # default_value

olr_wa_base_model_size0 = 1
olr_wa_base_model_size1 = 10
olr_wa_base_model_size2 = 2
olr_wa_increment_size = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features + 1) * 5))
olr_wa_increment_size2 = lambda n_features, user_defined_val: int(max(user_defined_val, (n_features ) * 5))