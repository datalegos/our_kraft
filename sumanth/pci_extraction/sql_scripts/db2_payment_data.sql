-- Payment Database Mock Data
-- Insert mock transactions with PCI sensitive information

INSERT INTO transactions (
    customer_id, merchant_id, transaction_type, amount, card_number_masked, 
    card_number_token, card_holder_name, authorization_code, processor_response_code, 
    processor_response_message, transaction_status, gateway_transaction_id, 
    risk_score, fraud_indicators, ip_address, user_agent, 
    billing_address_line1, billing_city, billing_state, billing_zip_code,
    processed_at
) VALUES
(1, 'MERCH_001', 'PURCHASE', 129.99, '****-****-****-9012', 'TKN_4532123456789012_001', 'John Doe', 'AUTH123456', '00', 'APPROVED', 'APPROVED', 'GW_TXN_001', 15, 'none', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', '123 Main St', 'New York', 'NY', '10001', '2024-01-15 10:30:00'),
(2, 'MERCH_002', 'PURCHASE', 89.50, '****-****-****-1111', 'TKN_4111111111111111_002', 'Jane Smith', 'AUTH789012', '00', 'APPROVED', 'APPROVED', 'GW_TXN_002', 8, 'none', '10.0.0.50', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', '456 Oak Ave', 'Los Angeles', 'CA', '90210', '2024-01-15 14:22:00'),
(3, 'MERCH_001', 'PURCHASE', 256.75, '****-****-***-10005', 'TKN_378282246310005_003', 'Michael Johnson', 'AUTH345678', '00', 'APPROVED', 'APPROVED', 'GW_TXN_003', 22, 'high_amount', '172.16.0.25', 'Mozilla/5.0 (X11; Linux x86_64)', '789 Pine Rd', 'Chicago', 'IL', '60601', '2024-01-16 09:15:00'),
(4, 'MERCH_003', 'PURCHASE', 45.00, '****-****-****-1117', 'TKN_6011111111111117_004', 'Sarah Williams', 'AUTH901234', '05', 'DECLINED', 'DECLINED', 'GW_TXN_004', 75, 'insufficient_funds,velocity_check', '203.0.113.10', 'Mozilla/5.0 (iPhone; CPU iPhone OS 15_0)', '321 Elm St', 'Houston', 'TX', '77001', '2024-01-16 16:45:00'),
(5, 'MERCH_002', 'PURCHASE', 199.99, '****-****-****-6666', 'TKN_4532888877776666_005', 'David Brown', 'AUTH567890', '00', 'APPROVED', 'APPROVED', 'GW_TXN_005', 12, 'none', '198.51.100.75', 'Mozilla/5.0 (Android 12; Mobile)', '654 Maple Dr', 'Phoenix', 'AZ', '85001', '2024-01-17 11:30:00'),
(1, 'MERCH_001', 'REFUND', -29.99, '****-****-****-9012', 'TKN_4532123456789012_001', 'John Doe', 'REF123456', '00', 'APPROVED', 'APPROVED', 'GW_TXN_006', 5, 'none', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', '123 Main St', 'New York', 'NY', '10001', '2024-01-17 15:20:00'),
(6, 'MERCH_004', 'PURCHASE', 75.25, '****-****-****-9999', 'TKN_5555123456789999_007', 'Emily Davis', 'AUTH234567', '00', 'APPROVED', 'APPROVED', 'GW_TXN_007', 18, 'new_customer', '203.0.113.50', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', '987 Cedar Ln', 'Philadelphia', 'PA', '19101', '2024-01-18 13:10:00'),
(7, 'MERCH_001', 'PURCHASE', 320.00, '****-****-****-4444', 'TKN_4111222233334444_008', 'Robert Miller', 'AUTH678901', '00', 'APPROVED', 'APPROVED', 'GW_TXN_008', 35, 'high_amount,international_card', '198.51.100.100', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', '147 Birch St', 'San Antonio', 'TX', '78201', '2024-01-18 20:45:00'),
(8, 'MERCH_002', 'PURCHASE', 67.80, '****-****-***-98431', 'TKN_371449635398431_009', 'Lisa Wilson', 'AUTH345612', '00', 'APPROVED', 'APPROVED', 'GW_TXN_009', 9, 'none', '10.0.0.200', 'Mozilla/5.0 (iPad; CPU OS 15_0)', '258 Spruce Ave', 'San Diego', 'CA', '92101', '2024-01-19 08:30:00');

-- Insert mock audit log entries for PCI compliance tracking
INSERT INTO payment_audit_log (
    transaction_id, action_type, user_id, user_role, ip_address, user_agent,
    sensitive_data_accessed, access_reason, compliance_notes, session_id
) VALUES
(1, 'CREATE', 'admin_001', 'ADMIN', '10.0.0.10', 'Internal System', 'card_number_token,card_holder_name', 'Transaction Processing', 'Standard transaction creation', 'SESS_001'),
(1, 'VIEW', 'analyst_001', 'ANALYST', '10.0.0.20', 'Mozilla/5.0 (Windows NT 10.0)', 'transaction_amount,status', 'Fraud Analysis', 'Routine fraud review', 'SESS_002'),
(2, 'CREATE', 'admin_001', 'ADMIN', '10.0.0.10', 'Internal System', 'card_number_token,card_holder_name', 'Transaction Processing', 'Standard transaction creation', 'SESS_003'),
(3, 'CREATE', 'admin_001', 'ADMIN', '10.0.0.10', 'Internal System', 'card_number_token,card_holder_name', 'Transaction Processing', 'High amount transaction flagged', 'SESS_004'),
(3, 'VIEW', 'security_001', 'SECURITY', '10.0.0.30', 'Mozilla/5.0 (X11; Linux x86_64)', 'fraud_indicators,risk_score', 'Security Review', 'High amount transaction review', 'SESS_005'),
(4, 'CREATE', 'admin_001', 'ADMIN', '10.0.0.10', 'Internal System', 'card_number_token,card_holder_name', 'Transaction Processing', 'Transaction declined by processor', 'SESS_006'),
(4, 'VIEW', 'support_001', 'SUPPORT', '10.0.0.40', 'Mozilla/5.0 (Windows NT 10.0)', 'processor_response_message', 'Customer Support', 'Customer inquiry about declined transaction', 'SESS_007'),
(5, 'CREATE', 'admin_001', 'ADMIN', '10.0.0.10', 'Internal System', 'card_number_token,card_holder_name', 'Transaction Processing', 'Standard transaction creation', 'SESS_008'),
(6, 'CREATE', 'admin_001', 'ADMIN', '10.0.0.10', 'Internal System', 'card_number_token,card_holder_name', 'Refund Processing', 'Customer refund request processed', 'SESS_009'),
(7, 'CREATE', 'admin_001', 'ADMIN', '10.0.0.10', 'Internal System', 'card_number_token,card_holder_name', 'Transaction Processing', 'New customer transaction', 'SESS_010');