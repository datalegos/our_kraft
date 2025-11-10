# UPI Payment Integration Guide

## 🇮🇳 What is UPI?

**UPI (Unified Payments Interface)** is India's instant payment system that allows you to transfer money between bank accounts instantly using a mobile app.

**Popular UPI Apps**:
- Google Pay (GPay)
- PhonePe
- Paytm
- BHIM
- Amazon Pay
- WhatsApp Pay

---

## 🔄 How UPI Payments Work

### Complete Flow

```
1. Customer on your website
   ↓
2. Clicks "Pay with UPI"
   ↓
3. Your backend creates payment intent
   ↓
4. Customer sees UPI payment options:
   - Enter UPI ID (user@paytm)
   - Scan QR code
   - Choose UPI app
   ↓
5. Customer selects Google Pay
   ↓
6. REDIRECTS to Google Pay app
   ↓
7. Google Pay shows payment details
   ↓
8. Customer enters UPI PIN
   ↓
9. Payment processed
   ↓
10. REDIRECTS back to your website
   ↓
11. Shows success message
```

---

## 💻 Implementation with Stripe

### Step 1: Backend - Create Payment Intent

```python
# app.py
import stripe

@app.route('/api/create-upi-payment', methods=['POST'])
def create_upi_payment():
    try:
        data = request.get_json()
        amount = data.get('amount')  # Amount in rupees
        
        # Convert to paise (smallest unit)
        amount_in_paise = int(float(amount) * 100)
        
        # Create Payment Intent with UPI
        intent = stripe.PaymentIntent.create(
            amount=amount_in_paise,
            currency='inr',
            payment_method_types=['upi'],  # Enable UPI
            metadata={
                'order_id': 'ORDER_123'
            }
        )
        
        return jsonify({
            'clientSecret': intent.client_secret,
            'paymentIntentId': intent.id
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

### Step 2: Frontend - Show UPI Options

```html
<!-- index.html -->
<div id="upi-payment-section">
    <h3>Pay with UPI</h3>
    
    <!-- Option 1: Enter UPI ID -->
    <div>
        <label>Enter UPI ID:</label>
        <input type="text" id="upi-id" placeholder="yourname@paytm">
        <button onclick="payWithUpiId()">Pay</button>
    </div>
    
    <!-- Option 2: Choose UPI App -->
    <div>
        <h4>Or choose your UPI app:</h4>
        <button onclick="payWithApp('gpay')">
            <img src="gpay-icon.png"> Google Pay
        </button>
        <button onclick="payWithApp('phonepe')">
            <img src="phonepe-icon.png"> PhonePe
        </button>
        <button onclick="payWithApp('paytm')">
            <img src="paytm-icon.png"> Paytm
        </button>
    </div>
    
    <!-- Option 3: QR Code -->
    <div>
        <h4>Or scan QR code:</h4>
        <div id="qr-code"></div>
    </div>
</div>
```

### Step 3: JavaScript - Handle UPI Payment

```javascript
// script.js

// Option 1: Pay with UPI ID
async function payWithUpiId() {
    const upiId = document.getElementById('upi-id').value;
    
    if (!upiId) {
        alert('Please enter UPI ID');
        return;
    }
    
    // Create payment intent
    const response = await fetch('/api/create-upi-payment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount: 100 })
    });
    
    const { clientSecret } = await response.json();
    
    // Confirm UPI payment
    const result = await stripe.confirmUpiPayment(clientSecret, {
        payment_method: {
            upi: {
                vpa: upiId  // UPI ID
            }
        },
        return_url: window.location.origin + '/payment-complete'
    });
    
    if (result.error) {
        alert('Payment failed: ' + result.error.message);
    }
    // If successful, user is redirected to UPI app
}

// Option 2: Pay with specific UPI app
async function payWithApp(appName) {
    // Create payment intent
    const response = await fetch('/api/create-upi-payment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount: 100 })
    });
    
    const { clientSecret } = await response.json();
    
    // Generate UPI deep link
    const upiLink = generateUpiDeepLink(appName, clientSecret);
    
    // Redirect to UPI app
    window.location.href = upiLink;
}

// Generate UPI deep link for specific app
function generateUpiDeepLink(app, clientSecret) {
    const baseUrls = {
        'gpay': 'tez://upi/pay',
        'phonepe': 'phonepe://pay',
        'paytm': 'paytmmp://pay'
    };
    
    const params = new URLSearchParams({
        pa: 'merchant@upi',  // Your UPI ID
        pn: 'Your Store',
        am: '100.00',
        cu: 'INR',
        tn: 'Payment for Order'
    });
    
    return `${baseUrls[app]}?${params.toString()}`;
}
```

---

## 📱 UPI Deep Links (App Redirect)

### What are Deep Links?

Deep links are special URLs that open specific apps on mobile devices.

### UPI Deep Link Format:

```
upi://pay?pa=merchant@upi&pn=StoreName&am=100&cu=INR&tn=Order123
```

**Parameters**:
- `pa` = Payee address (merchant UPI ID)
- `pn` = Payee name (your store name)
- `am` = Amount
- `cu` = Currency (INR)
- `tn` = Transaction note

### App-Specific Deep Links:

#### Google Pay
```
tez://upi/pay?pa=merchant@upi&pn=Store&am=100&cu=INR
```

#### PhonePe
```
phonepe://pay?pa=merchant@upi&pn=Store&am=100&cu=INR
```

#### Paytm
```
paytmmp://pay?pa=merchant@upi&pn=Store&am=100&cu=INR
```

#### BHIM
```
bhim://pay?pa=merchant@upi&pn=Store&am=100&cu=INR
```

### Implementation:

```javascript
function redirectToUpiApp(app) {
    const deepLinks = {
        gpay: 'tez://upi/pay?pa=yourstore@paytm&pn=YourStore&am=100&cu=INR',
        phonepe: 'phonepe://pay?pa=yourstore@paytm&pn=YourStore&am=100&cu=INR',
        paytm: 'paytmmp://pay?pa=yourstore@paytm&pn=YourStore&am=100&cu=INR'
    };
    
    // Redirect to app
    window.location.href = deepLinks[app];
    
    // Fallback: If app not installed, show QR code
    setTimeout(() => {
        showQRCode();
    }, 2000);
}
```

---

## 🔄 Complete User Journey

### Mobile Web Flow:

```
Step 1: Customer on your mobile website
┌─────────────────────────────────┐
│ 📱 Your Store                   │
│                                 │
│ Product: ₹100                   │
│                                 │
│ [Pay with UPI]                  │
└─────────────────────────────────┘

Step 2: Choose UPI app
┌─────────────────────────────────┐
│ Choose UPI App:                 │
│                                 │
│ [📱 Google Pay]  ← Tap          │
│ [📱 PhonePe]                    │
│ [📱 Paytm]                      │
└─────────────────────────────────┘

Step 3: Redirects to Google Pay
┌─────────────────────────────────┐
│ 🔵 Google Pay                   │
│                                 │
│ Pay to: Your Store              │
│ Amount: ₹100.00                 │
│                                 │
│ From: HDFC Bank •••• 1234       │
│                                 │
│ Enter UPI PIN:                  │
│ [• • • •]                       │
│                                 │
│ [Pay ₹100]                      │
└─────────────────────────────────┘

Step 4: Payment processing
┌─────────────────────────────────┐
│ Processing payment...           │
│ 🔄                              │
└─────────────────────────────────┘

Step 5: Success - Redirects back
┌─────────────────────────────────┐
│ ✅ Payment Successful!          │
│                                 │
│ Amount: ₹100.00                 │
│ Transaction ID: 123456789       │
│                                 │
│ [View Order]                    │
└─────────────────────────────────┘
```

---

## 🖥️ Desktop Flow (QR Code)

### For Desktop Users:

```
Step 1: Customer on desktop
┌─────────────────────────────────────┐
│ 💻 Your Store                       │
│                                     │
│ Product: ₹100                       │
│                                     │
│ [Pay with UPI]                      │
└─────────────────────────────────────┘

Step 2: Shows QR code
┌─────────────────────────────────────┐
│ Scan QR code with any UPI app:     │
│                                     │
│     ┌─────────────────┐             │
│     │  █▀▀▀▀▀█ ▀ █▀█  │             │
│     │  █ ███ █ ▄▀ ▀▄  │             │
│     │  █ ▀▀▀ █ █▀█▀█  │             │
│     │  ▀▀▀▀▀▀▀ ▀ ▀ ▀  │             │
│     └─────────────────┘             │
│                                     │
│ Or enter UPI ID:                    │
│ [yourname@paytm]                    │
└─────────────────────────────────────┘

Step 3: Customer scans with phone
┌─────────────────────────────────┐
│ 📱 Google Pay (on phone)        │
│                                 │
│ Scanned QR code                 │
│                                 │
│ Pay to: Your Store              │
│ Amount: ₹100.00                 │
│                                 │
│ [Pay ₹100]                      │
└─────────────────────────────────┘

Step 4: Desktop shows success
┌─────────────────────────────────────┐
│ ✅ Payment Successful!              │
│                                     │
│ Amount: ₹100.00                     │
│ Paid via: Google Pay                │
└─────────────────────────────────────┘
```

---

## 🔧 Implementation Options

### Option 1: Stripe UPI (Recommended)

**Pros**:
- ✅ Integrated with Stripe
- ✅ Automatic reconciliation
- ✅ Same dashboard as cards
- ✅ Webhook support

**Cons**:
- ⚠️ Stripe fees apply (2%)

**Code**:
```python
intent = stripe.PaymentIntent.create(
    amount=10000,
    currency='inr',
    payment_method_types=['upi']
)
```

---

### Option 2: Razorpay UPI

**Pros**:
- ✅ Lower fees (2%)
- ✅ India-focused
- ✅ Better UPI support

**Cons**:
- ⚠️ Separate integration

**Code**:
```python
order = razorpay_client.order.create({
    'amount': 10000,
    'currency': 'INR',
    'payment_capture': 1
})

# Frontend
var options = {
    key: 'rzp_test_...',
    amount: 10000,
    currency: 'INR',
    order_id: order['id'],
    method: 'upi',
    handler: function(response) {
        // Payment successful
    }
}
```

---

### Option 3: Direct UPI (Advanced)

**Pros**:
- ✅ No payment gateway fees
- ✅ Direct to your bank

**Cons**:
- ⚠️ Complex implementation
- ⚠️ Manual reconciliation
- ⚠️ Need UPI merchant account

**Code**:
```javascript
// Generate UPI payment link
const upiLink = `upi://pay?pa=yourstore@paytm&pn=YourStore&am=100&cu=INR&tn=Order123`;

// Redirect
window.location.href = upiLink;
```

---

## 📱 Detecting Mobile vs Desktop

```javascript
function isMobile() {
    return /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function showUpiPayment() {
    if (isMobile()) {
        // Show UPI app buttons
        showUpiAppButtons();
    } else {
        // Show QR code
        showQRCode();
    }
}
```

---

## 🔐 Security Considerations

### UPI Security Features:

```
✅ Two-factor authentication (UPI PIN)
✅ Encrypted transactions
✅ Bank-level security
✅ Transaction limits
✅ Instant notifications
```

### Best Practices:

```
✅ Verify payment on backend
✅ Use webhooks for confirmation
✅ Validate transaction IDs
✅ Log all transactions
✅ Handle timeouts gracefully
```

---

## ⏱️ Handling Timeouts

### UPI Payment Timeout:

```javascript
// Set timeout for UPI payment
const timeout = setTimeout(() => {
    // Check payment status
    checkPaymentStatus(paymentId);
}, 5 * 60 * 1000); // 5 minutes

async function checkPaymentStatus(paymentId) {
    const response = await fetch(`/api/check-payment/${paymentId}`);
    const data = await response.json();
    
    if (data.status === 'succeeded') {
        showSuccess();
    } else if (data.status === 'pending') {
        // Still processing
        showPending();
    } else {
        // Failed or cancelled
        showFailed();
    }
}
```

---

## 🎯 Complete Example

### Full Implementation:

```python
# Backend (app.py)
@app.route('/api/create-upi-payment', methods=['POST'])
def create_upi_payment():
    data = request.get_json()
    amount = int(float(data['amount']) * 100)
    
    intent = stripe.PaymentIntent.create(
        amount=amount,
        currency='inr',
        payment_method_types=['upi'],
        metadata={'order_id': data['order_id']}
    )
    
    return jsonify({
        'clientSecret': intent.client_secret,
        'paymentIntentId': intent.id
    })

@app.route('/payment-complete')
def payment_complete():
    payment_intent_id = request.args.get('payment_intent')
    
    # Verify payment
    intent = stripe.PaymentIntent.retrieve(payment_intent_id)
    
    if intent.status == 'succeeded':
        return render_template('success.html')
    else:
        return render_template('failed.html')
```

```javascript
// Frontend (script.js)
async function payWithUPI(appName) {
    // Create payment
    const response = await fetch('/api/create-upi-payment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            amount: 100,
            order_id: 'ORDER_123'
        })
    });
    
    const { clientSecret } = await response.json();
    
    // Redirect to UPI app
    const result = await stripe.confirmUpiPayment(clientSecret, {
        payment_method: {
            upi: {
                vpa: 'customer@upi'
            }
        },
        return_url: window.location.origin + '/payment-complete'
    });
    
    // User is redirected to UPI app
    // After payment, redirected back to return_url
}
```

---

## 📊 UPI vs Card Comparison

| Feature | UPI | Card |
|---------|-----|------|
| **Setup** | No setup needed | Enter card details |
| **Speed** | Instant | Instant |
| **Fees** | 2% | 2.9% + ₹0.30 |
| **Limit** | ₹1,00,000/day | No limit |
| **Refunds** | Instant | 5-7 days |
| **Popular in** | India | Global |

---

## ✅ Summary

### How UPI Redirect Works:

1. Customer clicks "Pay with UPI"
2. Chooses UPI app (GPay, PhonePe, etc.)
3. **Redirects to that app**
4. Customer approves in app
5. **Redirects back to your website**
6. Payment confirmed!

### Key Points:

- ✅ Works on mobile (app redirect)
- ✅ Works on desktop (QR code)
- ✅ Instant payments
- ✅ Lower fees than cards
- ✅ Popular in India

### Implementation:

- Use Stripe or Razorpay for UPI
- Handle mobile (deep links) and desktop (QR) differently
- Always verify payment on backend
- Use webhooks for reliability

---

**Ready to implement UPI?** Follow the code examples above! 🚀
