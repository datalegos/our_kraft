# Complete Setup Guide

## 📋 Prerequisites

- Python 3.7 or higher
- Internet connection
- Text editor

## 🔑 Step 1: Get Stripe Test Keys (5 minutes)

### Sign Up for Stripe

1. Go to https://stripe.com
2. Click "Sign up"
3. Fill in:
   - Email
   - Full name
   - Country (if India asks for invite, choose another country like US/UK)
   - Password
4. Click "Create account"

### Get Your Test Keys

1. After signup, look at **top right corner**
2. Find "Test mode" toggle - make sure it's **ON**
3. Click **"Developers"** in left sidebar
4. Click **"API keys"**
5. You'll see two keys:
   - **Publishable key**: `pk_test_...` (visible)
   - **Secret key**: `sk_test_...` (click "Reveal test key")
6. Copy both keys

## 💻 Step 2: Setup Project (5 minutes)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create .env File

```bash
copy .env.example .env
```

### Add Your Keys

Open `.env` file and add your keys:

```env
STRIPE_SECRET_KEY=sk_test_paste_your_full_key_here
STRIPE_PUBLISHABLE_KEY=pk_test_paste_your_full_key_here
```

**Important:**
- Copy the COMPLETE key (they're long)
- No spaces before/after the `=`
- Save the file

## ✅ Step 3: Verify Setup

Run the verification script:

```bash
python verify_setup.py
```

Expected output:
```
✅ Keys found in .env file
✅ Stripe library installed
✅ API connection successful!
✅ Test payment intent created!
🎉 SUCCESS!
```

## 🚀 Step 4: Run Application

```bash
python app.py
```

You should see:
```
✅ Available gateways: Stripe
🌐 Server running at: http://localhost:5000
```

## 🌐 Step 5: Test in Browser

1. Open: http://localhost:5000
2. Select: "Stripe (Test Mode - FREE)"
3. Amount: **10.00**
4. Currency: **USD**
5. Click: "Process Payment"
6. Card: **4242 4242 4242 4242**
7. Expiry: **12/25**
8. CVC: **123**
9. Submit

You should see: ✅ Payment successful!

## 🎴 Test Cards

### Success
```
Card: 4242 4242 4242 4242
Expiry: Any future date
CVC: Any 3 digits
```

### Declined
```
Card: 4000 0000 0000 0002
```

### Insufficient Funds
```
Card: 4000 0000 0000 9995
```

### 3D Secure
```
Card: 4000 0027 6000 3184
```

More: https://stripe.com/docs/testing

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
pip install -r requirements.txt
```

### Issue: "Stripe is not configured"
1. Check `.env` file exists
2. Verify keys start with `sk_test_` and `pk_test_`
3. Restart Flask: Ctrl+C, then `python app.py`

### Issue: "Invalid API key"
1. Make sure you're in Test mode (not Live mode)
2. Copy keys again from Stripe dashboard
3. Check for extra spaces in `.env`

### Issue: "Amount too small"
- Minimum: $0.50 USD or ₹50 INR
- Use: $10.00 USD for testing

### Issue: "Payment fails"
- Use test card: 4242 4242 4242 4242
- Check expiry is in future
- Test mode only accepts test cards

## 📚 Additional Resources

- **Stripe Dashboard**: https://dashboard.stripe.com/test/dashboard
- **API Keys**: https://dashboard.stripe.com/test/apikeys
- **Documentation**: https://stripe.com/docs
- **Test Cards**: https://stripe.com/docs/testing
- **Support**: https://support.stripe.com

## ✅ Success Checklist

- [ ] Signed up for Stripe
- [ ] Got test API keys
- [ ] Installed dependencies
- [ ] Created .env file
- [ ] Added keys to .env
- [ ] Ran verify_setup.py successfully
- [ ] Started Flask app
- [ ] Opened http://localhost:5000
- [ ] Processed test payment successfully

## 🎉 You're Done!

Your payment gateway POC is now running with real Stripe integration!

**Next Steps:**
- Try different test cards
- Click "Explore API Calls" to see detailed logs
- View payments in Stripe dashboard
- Modify the code to add features
