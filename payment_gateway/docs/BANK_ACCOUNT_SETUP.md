# Bank Account Setup Guide

## 💰 Where Does the Money Go?

When customers pay you through Stripe, the money goes to **YOUR bank account** that you configure in Stripe Dashboard.

---

## 🏦 Setting Up Your Bank Account (Step-by-Step)

### Step 1: Complete Stripe Account Activation

Before you can add a bank account, you need to activate your Stripe account.

**Go to**: https://dashboard.stripe.com

**You'll see a banner**: "Activate your account to start accepting live payments"

**Click**: "Activate account" or "Complete account setup"

---

### Step 2: Provide Business Information

Stripe will ask for:

#### Personal/Business Details
```
✅ Business type (Individual, Company, Non-profit)
✅ Business name
✅ Business address
✅ Business website (if you have one)
✅ Business description
✅ Industry/Category
```

#### Personal Information
```
✅ Your full legal name
✅ Date of birth
✅ Home address
✅ Phone number
✅ Email address
✅ Tax ID / SSN (for US) or equivalent
```

#### Identity Verification
```
✅ Government-issued ID (Passport, Driver's License)
✅ Photo of yourself (sometimes required)
✅ Proof of address (utility bill, bank statement)
```

---

### Step 3: Add Your Bank Account

#### For US Bank Accounts:

1. **Go to**: Settings → Bank accounts and scheduling
2. **Click**: "Add bank account"
3. **Enter**:
   ```
   Bank name: [Your Bank Name]
   Account holder name: [Your Name]
   Routing number: [9 digits]
   Account number: [Your account number]
   Account type: Checking or Savings
   ```

4. **Verify**: Stripe will send 2 small deposits (like $0.32 and $0.45)
5. **Confirm**: Enter the amounts to verify ownership

#### For Indian Bank Accounts:

1. **Go to**: Settings → Bank accounts and scheduling
2. **Click**: "Add bank account"
3. **Enter**:
   ```
   Account holder name: [Your Name]
   Account number: [Your account number]
   IFSC code: [Bank IFSC code]
   Account type: Savings or Current
   ```

4. **Verify**: May require additional documents
5. **Confirm**: Follow verification steps

#### For Other Countries:

1. **Go to**: Settings → Bank accounts and scheduling
2. **Click**: "Add bank account"
3. **Enter**: IBAN or local bank details
4. **Verify**: Follow country-specific verification

---

## 💸 Payout Schedule

### How Often You Get Paid

**Default Schedule**:
- **Daily**: Money arrives next business day
- **Weekly**: Money arrives once per week
- **Monthly**: Money arrives once per month

**Configure in**: Settings → Bank accounts and scheduling → Payout schedule

### Example Timeline:

```
Monday: Customer pays $100
   ↓
Tuesday: Stripe processes payment
   ↓
Wednesday: Money arrives in your bank account
```

**Note**: First payout may take 7-14 days for security verification.

---

## 💰 How Much Money You Receive

### Stripe Fees

**Standard Pricing** (US):
```
2.9% + $0.30 per successful card charge
```

**Example**:
```
Customer pays: $100.00
Stripe fee: $3.20 (2.9% + $0.30)
You receive: $96.80
```

**International Cards**:
```
3.9% + $0.30 per charge
```

**Currency Conversion**:
```
Additional 1% for currency conversion
```

### Fee Breakdown by Country

#### United States
```
Domestic cards: 2.9% + $0.30
International cards: 3.9% + $0.30
```

#### India
```
Domestic cards: 2% + ₹0
International cards: 3% + ₹0
UPI: 2%
```

#### Europe
```
European cards: 1.4% + €0.25
UK cards: 1.4% + £0.20
International cards: 2.9% + €0.25
```

---

## 🔍 Viewing Your Balance

### Check Your Balance

1. **Go to**: Stripe Dashboard → Balance
2. **You'll see**:
   ```
   Available balance: $500.00
   Pending balance: $200.00
   ```

**Available**: Ready to be paid out
**Pending**: Being processed (usually 2-7 days)

### Payout History

1. **Go to**: Stripe Dashboard → Balance → Payouts
2. **You'll see**:
   ```
   Date        Amount      Status      Bank
   Nov 9       $500.00     Paid        •••• 1234
   Nov 8       $300.00     Paid        •••• 1234
   Nov 7       $450.00     In transit  •••• 1234
   ```

---

## 🏦 Multiple Bank Accounts

### Can You Add Multiple Banks?

**Yes!** You can add multiple bank accounts.

**Use Cases**:
- Different accounts for different currencies
- Backup account
- Business vs Personal

**How to Add**:
1. Go to: Settings → Bank accounts
2. Click: "Add bank account"
3. Set as default or use for specific purposes

---

## 🌍 International Payouts

### Receiving Money in Different Countries

**Stripe Atlas** (for US company):
- Get a US bank account
- Receive payouts in USD
- Available globally

**Local Bank Accounts**:
- Add bank account in your country
- Receive in local currency
- Stripe handles conversion

**Wise (formerly TransferWise)**:
- Get multi-currency account
- Lower conversion fees
- Works with Stripe

---

## 🔒 Security & Verification

### Why Verification is Required

**Stripe needs to verify**:
- ✅ You are who you say you are
- ✅ The bank account belongs to you
- ✅ You're not committing fraud
- ✅ Compliance with regulations

### What Stripe Checks

```
✅ Identity verification (ID, passport)
✅ Address verification (utility bill)
✅ Bank account ownership (micro-deposits)
✅ Business legitimacy (website, documents)
✅ Tax compliance (Tax ID, SSN)
```

### Verification Timeline

```
Day 1: Submit information
Day 2-3: Stripe reviews
Day 3-5: Micro-deposits sent
Day 5-7: You verify deposits
Day 7: Account fully activated
```

---

## 💡 Common Issues & Solutions

### Issue 1: "Bank account verification failed"

**Causes**:
- Wrong routing/account number
- Account holder name doesn't match
- Account closed or frozen

**Solution**:
1. Double-check all details
2. Contact your bank to confirm account is active
3. Try adding account again
4. Contact Stripe support if still failing

---

### Issue 2: "Payouts are paused"

**Causes**:
- Unusual activity detected
- Verification incomplete
- High chargeback rate
- Suspicious transactions

**Solution**:
1. Check email from Stripe
2. Complete any pending verification
3. Respond to Stripe's requests
4. Contact Stripe support

---

### Issue 3: "First payout is delayed"

**This is normal!**

**Why**:
- Stripe holds first payout for 7-14 days
- Security measure for new accounts
- Protects against fraud

**What to do**:
- Wait for the initial period
- Future payouts will be faster
- Build transaction history

---

### Issue 4: "Can't add bank account"

**Causes**:
- Bank not supported
- Account type not supported
- Country restrictions

**Solution**:
1. Check if your bank is supported
2. Try a different bank
3. Use alternative like Wise
4. Contact Stripe support

---

## 📊 Payout Reports

### Viewing Detailed Reports

1. **Go to**: Reports → Balance
2. **Download**: CSV or PDF
3. **See**:
   - All transactions
   - Fees charged
   - Net payouts
   - Refunds
   - Chargebacks

### Tax Reporting

**US Users**:
- Stripe provides 1099-K form
- Sent if you process >$20,000 and >200 transactions
- Available in Dashboard → Reports → Tax documents

**Other Countries**:
- Download transaction reports
- Provide to your accountant
- Use for tax filing

---

## 🎯 Best Practices

### 1. Verify Bank Details Carefully
```
✅ Double-check routing number
✅ Confirm account number
✅ Ensure name matches exactly
✅ Use business account for business
```

### 2. Set Up Notifications
```
✅ Enable payout notifications
✅ Get alerts for failed payouts
✅ Monitor balance regularly
```

### 3. Keep Information Updated
```
✅ Update address if you move
✅ Update bank if you change banks
✅ Keep contact info current
```

### 4. Monitor Your Balance
```
✅ Check balance weekly
✅ Reconcile with your records
✅ Report discrepancies immediately
```

---

## 🔐 Security Tips

### Protect Your Bank Account

```
❌ Never share bank login credentials
❌ Don't use public WiFi for banking
❌ Don't share Stripe dashboard access
✅ Use strong passwords
✅ Enable 2-factor authentication
✅ Monitor for suspicious activity
```

### Stripe Security Features

```
✅ Encrypted data transmission
✅ PCI DSS Level 1 certified
✅ Fraud detection
✅ Secure bank connections
✅ Regular security audits
```

---

## 📞 Getting Help

### Stripe Support

**Email**: support@stripe.com
**Phone**: Available in Dashboard
**Chat**: Dashboard → Help → Contact support
**Docs**: https://stripe.com/docs

### Bank Issues

**Contact your bank**:
- Verify account is active
- Confirm account details
- Check for any holds or restrictions

---

## ✅ Checklist: Bank Account Setup

- [ ] Stripe account activated
- [ ] Business information provided
- [ ] Identity verified
- [ ] Bank account added
- [ ] Bank account verified (micro-deposits)
- [ ] Payout schedule configured
- [ ] Test payout received
- [ ] Notifications enabled
- [ ] Tax information provided
- [ ] Ready to receive payments!

---

## 🎯 Summary

### Key Points:

1. **You need a bank account** to receive money from Stripe
2. **Verification takes 7-14 days** for first payout
3. **Stripe fees are deducted** before payout
4. **Payouts are automatic** based on your schedule
5. **You can change banks** anytime in settings

### Money Flow:

```
Customer pays $100
    ↓
Stripe receives $100
    ↓
Stripe deducts fee ($3.20)
    ↓
Your balance: $96.80
    ↓
Payout to your bank (next day)
    ↓
Your bank account: +$96.80
```

---

## 🚀 Next Steps

1. **Complete Stripe activation**
2. **Add your bank account**
3. **Verify ownership**
4. **Wait for first payout**
5. **Start accepting payments!**

---

**Need help?** Contact Stripe support or check their documentation at https://stripe.com/docs/payouts
