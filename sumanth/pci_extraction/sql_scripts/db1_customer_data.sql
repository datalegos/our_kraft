-- Customer Database Mock Data
-- Insert mock customers with PCI sensitive information

INSERT INTO customers (first_name, last_name, email, phone, ssn, date_of_birth, address_line1, city, state, zip_code) VALUES
('John', 'Doe', 'john.doe@email.com', '555-0101', '123-45-6789', '1985-03-15', '123 Main St', 'New York', 'NY', '10001'),
('Jane', 'Smith', 'jane.smith@email.com', '555-0102', '987-65-4321', '1990-07-22', '456 Oak Ave', 'Los Angeles', 'CA', '90210'),
('Michael', 'Johnson', 'michael.j@email.com', '555-0103', '456-78-9123', '1988-11-08', '789 Pine Rd', 'Chicago', 'IL', '60601'),
('Sarah', 'Williams', 'sarah.w@email.com', '555-0104', '321-54-9876', '1992-02-14', '321 Elm St', 'Houston', 'TX', '77001'),
('David', 'Brown', 'david.brown@email.com', '555-0105', '654-32-1098', '1987-09-30', '654 Maple Dr', 'Phoenix', 'AZ', '85001'),
('Emily', 'Davis', 'emily.davis@email.com', '555-0106', '789-12-3456', '1991-12-05', '987 Cedar Ln', 'Philadelphia', 'PA', '19101'),
('Robert', 'Miller', 'robert.m@email.com', '555-0107', '147-25-8369', '1986-06-18', '147 Birch St', 'San Antonio', 'TX', '78201'),
('Lisa', 'Wilson', 'lisa.wilson@email.com', '555-0108', '258-36-9147', '1989-04-27', '258 Spruce Ave', 'San Diego', 'CA', '92101');

-- Insert mock payment methods with PCI sensitive information
INSERT INTO customer_payment_methods (customer_id, card_number, card_holder_name, expiry_month, expiry_year, cvv, card_type, billing_address_line1, billing_city, billing_state, billing_zip_code, is_primary) VALUES
(1, '4532-1234-5678-9012', 'John Doe', 12, 2026, '123', 'VISA', '123 Main St', 'New York', 'NY', '10001', TRUE),
(1, '5555-4444-3333-2222', 'John Doe', 08, 2025, '456', 'MASTERCARD', '123 Main St', 'New York', 'NY', '10001', FALSE),
(2, '4111-1111-1111-1111', 'Jane Smith', 03, 2027, '789', 'VISA', '456 Oak Ave', 'Los Angeles', 'CA', '90210', TRUE),
(3, '3782-822463-10005', 'Michael Johnson', 11, 2025, '1234', 'AMEX', '789 Pine Rd', 'Chicago', 'IL', '60601', TRUE),
(4, '6011-1111-1111-1117', 'Sarah Williams', 07, 2026, '567', 'DISCOVER', '321 Elm St', 'Houston', 'TX', '77001', TRUE),
(5, '4532-8888-7777-6666', 'David Brown', 09, 2025, '890', 'VISA', '654 Maple Dr', 'Phoenix', 'AZ', '85001', TRUE),
(6, '5555-1234-5678-9999', 'Emily Davis', 01, 2028, '234', 'MASTERCARD', '987 Cedar Ln', 'Philadelphia', 'PA', '19101', TRUE),
(7, '4111-2222-3333-4444', 'Robert Miller', 05, 2026, '678', 'VISA', '147 Birch St', 'San Antonio', 'TX', '78201', TRUE),
(8, '3714-496353-98431', 'Lisa Wilson', 10, 2027, '9012', 'AMEX', '258 Spruce Ave', 'San Diego', 'CA', '92101', TRUE);