-- Customer Database Schema
-- Database 1: Customer Information with PCI Data

-- Drop tables if they exist
DROP TABLE IF EXISTS customer_payment_methods;
DROP TABLE IF EXISTS customers;

-- Customers table with PCI sensitive information
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    ssn VARCHAR(11), -- PCI sensitive data
    date_of_birth DATE,
    address_line1 VARCHAR(100),
    address_line2 VARCHAR(100),
    city VARCHAR(50),
    state VARCHAR(50),
    zip_code VARCHAR(10),
    country VARCHAR(50) DEFAULT 'USA',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customer Payment Methods table with PCI sensitive information
CREATE TABLE customer_payment_methods (
    payment_method_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    card_number VARCHAR(19), -- PCI sensitive data (encrypted in real scenarios)
    card_holder_name VARCHAR(100),
    expiry_month INTEGER,
    expiry_year INTEGER,
    cvv VARCHAR(4), -- PCI sensitive data
    card_type VARCHAR(20), -- VISA, MASTERCARD, AMEX, etc.
    billing_address_line1 VARCHAR(100),
    billing_address_line2 VARCHAR(100),
    billing_city VARCHAR(50),
    billing_state VARCHAR(50),
    billing_zip_code VARCHAR(10),
    billing_country VARCHAR(50) DEFAULT 'USA',
    is_primary BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_ssn ON customers(ssn);
CREATE INDEX idx_payment_methods_customer_id ON customer_payment_methods(customer_id);
CREATE INDEX idx_payment_methods_card_number ON customer_payment_methods(card_number);