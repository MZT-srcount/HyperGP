import pyotp

key = 'YNE4L5QWC4NVROEFVOBHKE5JAZ35VDJP'
totp = pyotp.TOTP(key)
print(totp.now())
