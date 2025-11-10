# Payment Gateway POC

A Flask-based payment gateway integration demonstrating Stripe payment processing with test APIs.

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Stripe account (free signup at https://stripe.com)

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   copy .env.example .env
   ```
   
3. **Add your Stripe test keys to `.env`:**
   ```env
   STRIPE_SECRET_KEY=sk_test_your_key_here
   STRIPE_PUBLISHABLE_KEY=pk_test_your_key_here
   ```

4. **Get Stripe test keys:**
   - Sign up at https://stripe.com
   - Go to Dashboard → Developers → API Keys
   - Toggle "Test mode" ON
   - Copy both keys

5. **Run the application:**
   ```bash
   python app.py
   ```

6. **Open browser:**
   ```
   http://localhost:5000
   ```

## 💳 Test Cards

### Success
```
Card: 4242 4242 4242 4242
Expiry: 12/25
CVC: 123
Amount: $10.00 USD (minimum $0.50)
```

### Declined
```
Card: 4000 0000 0000 0002
```

### More test cards: https://stripe.com/docs/testing

## 📁 Project Structure

```
payment-gateway-poc/
├── app.py              # Flask backend
├── requirements.txt    # Dependencies
├── .env               # Your API keys (not in git)
├── .env.example       # Template for .env
├── templates/         # HTML templates
│   └── index.html
├── static/            # CSS and JavaScript
│   ├── style.css
│   └── script.js
└── docs/              # Documentation
    └── SETUP_GUIDE.md
```

## 🔧 Features

- ✅ Stripe payment integration
- ✅ Real-time API logging
- ✅ Test mode (no real money)
- ✅ Multiple test scenarios
- ✅ Clean, modern UI

## 📚 Documentation

### Setup & Testing
- **[SETUP_GUIDE.md](docs/SETUP_GUIDE.md)** - Detailed setup instructions
- **[Test Cards](https://stripe.com/docs/testing)** - All test card numbers

### Production & Deployment
- **[PRODUCTION_GUIDE.md](docs/PRODUCTION_GUIDE.md)** - How real payments work
- **[PAYMENT_FLOW_DIAGRAM.md](docs/PAYMENT_FLOW_DIAGRAM.md)** - Visual payment flows
- **[BANK_ACCOUNT_SETUP.md](docs/BANK_ACCOUNT_SETUP.md)** - Setting up payouts
- **[UPI_INTEGRATION.md](docs/UPI_INTEGRATION.md)** - UPI payments & app redirects

### External Resources
- **[Stripe Docs](https://stripe.com/docs)** - Official Stripe documentation
- **[Going Live Checklist](https://stripe.com/docs/development/checklist)** - Production deployment

## 🔒 Security Notes

- Never commit `.env` file
- Use test mode for development
- Test keys start with `sk_test_` and `pk_test_`
- No real money in test mode

## 🆘 Troubleshooting

### "Stripe is not configured"
- Check `.env` file exists
- Verify keys are correct
- Restart Flask app

### "Amount too small"
- Use minimum $0.50 USD or ₹50 INR

### "Invalid card"
- Use test cards only in test mode
- Card: 4242 4242 4242 4242

## 📄 License

MIT License - Free for learning and commercial use
