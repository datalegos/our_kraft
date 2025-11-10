#!/usr/bin/env python3
"""
Quick script to test if your Stripe test keys work
Run this after adding keys to .env file
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_stripe_keys():
    """Test if Stripe keys are configured and working"""
    
    print("\n" + "="*60)
    print("  Testing Stripe Test Keys")
    print("="*60 + "\n")
    
    # Check if keys exist
    secret_key = os.getenv('STRIPE_SECRET_KEY', '')
    publishable_key = os.getenv('STRIPE_PUBLISHABLE_KEY', '')
    
    if not secret_key:
        print("❌ STRIPE_SECRET_KEY not found in .env file")
        print("\n📝 To fix:")
        print("1. Go to https://dashboard.stripe.com/test/apikeys")
        print("2. Make sure 'Test mode' is ON (top right)")
        print("3. Copy your Secret key (starts with sk_test_)")
        print("4. Add to .env file: STRIPE_SECRET_KEY=sk_test_...")
        return False
    
    if not publishable_key:
        print("❌ STRIPE_PUBLISHABLE_KEY not found in .env file")
        print("\n📝 To fix:")
        print("1. Go to https://dashboard.stripe.com/test/apikeys")
        print("2. Make sure 'Test mode' is ON (top right)")
        print("3. Copy your Publishable key (starts with pk_test_)")
        print("4. Add to .env file: STRIPE_PUBLISHABLE_KEY=pk_test_...")
        return False
    
    # Check key format
    if not secret_key.startswith('sk_test_'):
        print("⚠️  WARNING: Secret key doesn't start with 'sk_test_'")
        print("   You might be using a LIVE key instead of TEST key!")
        print("   Make sure 'Test mode' is ON in Stripe dashboard")
        return False
    
    if not publishable_key.startswith('pk_test_'):
        print("⚠️  WARNING: Publishable key doesn't start with 'pk_test_'")
        print("   You might be using a LIVE key instead of TEST key!")
        print("   Make sure 'Test mode' is ON in Stripe dashboard")
        return False
    
    print("✅ Keys found in .env file")
    print(f"   Secret key: {secret_key[:15]}...{secret_key[-4:]}")
    print(f"   Publishable key: {publishable_key[:15]}...{publishable_key[-4:]}")
    
    # Try to import and use Stripe
    try:
        import stripe
        print("\n✅ Stripe library installed")
    except ImportError:
        print("\n❌ Stripe library not installed")
        print("   Run: pip install stripe")
        return False
    
    # Test API connection
    try:
        stripe.api_key = secret_key
        print("\n🔄 Testing API connection...")
        
        # Try to list payment intents (should work even if empty)
        result = stripe.PaymentIntent.list(limit=1)
        
        print("✅ API connection successful!")
        print(f"   Connected to Stripe API")
        print(f"   Test mode: Active")
        
        # Try to create a test payment intent
        print("\n🔄 Creating test payment intent...")
        test_intent = stripe.PaymentIntent.create(
            amount=1000,  # $10.00
            currency='usd',
            metadata={'test': 'verification'}
        )
        
        print("✅ Test payment intent created!")
        print(f"   Payment Intent ID: {test_intent.id}")
        print(f"   Amount: ${test_intent.amount / 100:.2f}")
        print(f"   Status: {test_intent.status}")
        
        print("\n" + "="*60)
        print("  🎉 SUCCESS! Your Stripe test keys work perfectly!")
        print("="*60)
        print("\n✅ You can now:")
        print("   1. Run: python app.py")
        print("   2. Open: http://localhost:5000")
        print("   3. Select 'Stripe' from dropdown")
        print("   4. Test with card: 4242 4242 4242 4242")
        print("\n")
        
        return True
        
    except stripe.error.AuthenticationError as e:
        print(f"\n❌ Authentication failed: {e}")
        print("\n📝 This means:")
        print("   - Your API key might be incorrect")
        print("   - Copy the key again from Stripe dashboard")
        print("   - Make sure there are no extra spaces")
        print("   - Make sure you're in TEST mode")
        return False
        
    except stripe.error.StripeError as e:
        print(f"\n❌ Stripe API error: {e}")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_stripe_keys()
    
    if not success:
        print("\n💡 Need help?")
        print("   1. Make sure you signed up at https://stripe.com")
        print("   2. Toggle to 'Test mode' (top right in dashboard)")
        print("   3. Go to Developers → API keys")
        print("   4. Copy BOTH keys to .env file")
        print("   5. Run this script again")
        print("\n📚 See SETUP_GUIDE.md for detailed instructions")
