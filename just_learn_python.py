resume_path = ''

if resume_path:
    print("This will NOT be printed.")
else:
    print("This WILL be printed, because an empty string is False.")

resume_path_valid = '/path/to/checkpoint.ckpt'

if resume_path_valid:
    print("This WILL be printed, because a non-empty string is True.")
else:
    print("This will NOT be printed.")