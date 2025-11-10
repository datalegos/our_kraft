// Initialize Stripe
let stripe = null;
if (STRIPE_PUBLISHABLE_KEY) {
    stripe = Stripe(STRIPE_PUBLISHABLE_KEY);
}

// Add log entry
function addLogEntry(type, title, data) {
    const logContent = document.getElementById('apiLog');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    
    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    entry.appendChild(titleEl);
    
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(data, null, 2);
    entry.appendChild(pre);
    
    // Remove info message if exists
    const infoMsg = logContent.querySelector('.log-info');
    if (infoMsg) {
        infoMsg.remove();
    }
    
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}

// Clear log
document.getElementById('clearLog').addEventListener('click', function() {
    document.getElementById('apiLog').innerHTML = '<p class="log-info">Log cleared</p>';
});

// Show payment status
function showPaymentStatus(status, message, data = null) {
    const statusDiv = document.getElementById('paymentStatus');
    const contentDiv = document.getElementById('statusContent');
    
    statusDiv.style.display = 'block';
    
    if (status === 'success') {
        contentDiv.className = 'status-success';
        contentDiv.innerHTML = `<strong>✅ ${message}</strong>`;
        if (data) {
            contentDiv.innerHTML += `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }
    } else {
        contentDiv.className = 'status-error';
        contentDiv.innerHTML = `<strong>❌ ${message}</strong>`;
        if (data) {
            contentDiv.innerHTML += `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }
    }
}

// Process Stripe Payment
async function processStripePayment(amount, currency) {
    addLogEntry('request', 'Step 1: Creating Stripe Payment Intent', {
        endpoint: '/api/stripe/create-payment-intent',
        method: 'POST',
        body: { amount, currency }
    });
    
    const response = await fetch('/api/stripe/create-payment-intent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ amount, currency })
    });
    
    const data = await response.json();
    addLogEntry('response', 'Step 1: Payment Intent Created', data);
    
    if (!response.ok) throw new Error(data.error || 'Failed to create payment intent');
    
    // Step 2: Confirm payment with Stripe Elements
    addLogEntry('request', 'Step 2: Confirming Payment with Stripe', {
        action: 'stripe.confirmCardPayment',
        clientSecret: data.clientSecret
    });
    
    const { error, paymentIntent } = await stripe.confirmCardPayment(data.clientSecret, {
        payment_method: {
            card: {
                token: 'tok_visa' // Test token for demo
            }
        }
    });
    
    if (error) {
        addLogEntry('error', 'Step 2: Payment Failed', { error: error.message });
        throw new Error(error.message);
    }
    
    addLogEntry('response', 'Step 2: Payment Confirmed', paymentIntent);
    
    // Step 3: Retrieve payment details
    addLogEntry('request', 'Step 3: Retrieving Payment Intent', {
        endpoint: `/api/stripe/retrieve-payment-intent/${paymentIntent.id}`,
        method: 'GET'
    });
    
    const retrieveResponse = await fetch(`/api/stripe/retrieve-payment-intent/${paymentIntent.id}`);
    const retrieveData = await retrieveResponse.json();
    addLogEntry('response', 'Step 3: Payment Details Retrieved', retrieveData);
    
    return retrieveData;
}

// Explore API Calls button
document.getElementById('exploreAPI').addEventListener('click', async function() {
    const amount = parseFloat(document.getElementById('amount').value);
    const currency = document.getElementById('currency').value;
    
    if (!amount) {
        alert('Please enter an amount');
        return;
    }
    
    if (!stripe) {
        alert('Stripe is not configured. Please add your Stripe keys to .env file');
        return;
    }
    
    try {
        const result = await processStripePayment(amount, currency);
        showPaymentStatus('success', 'Stripe payment processed successfully!', result);
    } catch (error) {
        addLogEntry('error', 'Error occurred', { error: error.message });
        showPaymentStatus('error', error.message);
    }
});

// Process Payment button
document.getElementById('processPayment').addEventListener('click', async function() {
    const button = this;
    button.disabled = true;
    button.textContent = 'Processing...';
    
    const amount = parseFloat(document.getElementById('amount').value);
    const currency = document.getElementById('currency').value;
    
    if (!amount) {
        alert('Please enter an amount');
        button.disabled = false;
        button.textContent = 'Process Payment';
        return;
    }
    
    if (!stripe) {
        alert('Stripe is not configured. Please add your Stripe keys to .env file');
        button.disabled = false;
        button.textContent = 'Process Payment';
        return;
    }
    
    try {
        const result = await processStripePayment(amount, currency);
        showPaymentStatus('success', 'Payment processed successfully!', result);
    } catch (error) {
        showPaymentStatus('error', error.message);
    } finally {
        button.disabled = false;
        button.textContent = 'Process Payment';
    }
});
