-- Payment Database Schema
-- Database 2: Transaction and Payment Processing with PCI Data

-- Drop tables if they exist
DROP TABLE IF EXISTS payment_audit_log;
DROP TABLE IF EXISTS transactions;

-- Transactions table with PCI sensitive information
CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    merchant_id VARCHAR(50),
    transaction_type VARCHAR(20) NOT NULL, -- PURCHASE, REFUND, VOID
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    card_number_masked VARCHAR(19), -- Masked card number (e.g., ****-****-****-1234)
    card_number_token VARCHAR(100), -- Tokenized card number for PCI compliance
    card_holder_name VARCHAR(100),
    authorization_code VARCHAR(20),
    processor_response_code VARCHAR(10),
    processor_response_message VARCHAR(255),
    transaction_status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, APPROVED, DECLINED, FAILED
    gateway_transaction_id VARCHAR(100),
    risk_score INTEGER,
    fraud_indicators TEXT, -- JSON or comma-separated fraud indicators
    ip_address INET,
    user_agent TEXT,
    billing_address_line1 VARCHAR(100),
    billing_city VARCHAR(50),
    billing_state VARCHAR(50),
    billing_zip_code VARCHAR(10),
    shipping_address_line1 VARCHAR(100),
    shipping_city VARCHAR(50),
    shipping_state VARCHAR(50),
    shipping_zip_code VARCHAR(10),
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Payment Audit Log table for PCI compliance tracking
CREATE TABLE payment_audit_log (
    audit_id SERIAL PRIMARY KEY,
    transaction_id INTEGER REFERENCES transactions(transaction_id),
    action_type VARCHAR(50) NOT NULL, -- CREATE, UPDATE, DELETE, VIEW, EXPORT
    user_id VARCHAR(50),
    user_role VARCHAR(50),
    ip_address INET,
    user_agent TEXT,
    sensitive_data_accessed TEXT, -- Which PCI fields were accessed
    access_reason VARCHAR(255),
    compliance_notes TEXT,
    session_id VARCHAR(100),
    audit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance and compliance reporting
CREATE INDEX idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX idx_transactions_card_token ON transactions(card_number_token);
CREATE INDEX idx_transactions_status ON transactions(transaction_status);
CREATE INDEX idx_transactions_processed_at ON transactions(processed_at);
CREATE INDEX idx_audit_log_transaction_id ON payment_audit_log(transaction_id);
CREATE INDEX idx_audit_log_user_id ON payment_audit_log(user_id);
CREATE INDEX idx_audit_log_timestamp ON payment_audit_log(audit_timestamp);
CREATE INDEX idx_audit_log_action_type ON payment_audit_log(action_type);