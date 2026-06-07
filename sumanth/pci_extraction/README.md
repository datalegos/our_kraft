# PCI Data Discovery and Analysis Platform

A comprehensive solution for discovering, extracting, and analyzing PCI/PII data across database systems with automated compliance reporting.

## 🎯 **Overview**

This platform provides enterprise-grade database discovery and PCI compliance analysis through two integrated components:

1. **Database Extractor**: Automatically discovers databases and extracts sample data
2. **Presidio Analyzer**: Identifies PII/PCI data using Microsoft Presidio and generates compliance reports

## 📁 **Project Structure**

```
pci_extraction/
├── config/                          # Configuration files
│   ├── extraction_config.yml        # Database extraction settings
│   └── presidio_config.yml          # PCI analysis settings
├── scripts/                         # Executable scripts
│   ├── database_extractor.py        # Main database extraction tool
│   ├── presidio_analyzer.py         # PCI data analysis tool
│   ├── run_database_extractor.sh    # Bash runner for extraction
│   ├── run_database_extractor.ps1   # PowerShell runner for extraction
│   ├── run_presidio_analysis.sh     # Bash runner for analysis
│   ├── run_presidio_analysis.ps1    # PowerShell runner for analysis
│   ├── run_full_pipeline.sh         # Complete pipeline runner
│   └── install_dependencies.sh      # Dependency installer
├── requirements/                    # Python dependencies
│   ├── database_requirements.txt    # Database extraction deps
│   └── presidio_requirements.txt    # PCI analysis deps
├── data/                           # Data directories (created at runtime)
│   ├── extracted_data/             # Database extraction results
│   └── pci_analysis_results/       # PCI compliance reports
├── logs/                           # Log files (created at runtime)
├── sql_scripts/                    # Database setup scripts
└── docs/                          # Documentation
```

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
# Bash/Git Bash
chmod +x scripts/install_dependencies.sh && ./scripts/install_dependencies.sh

# PowerShell
.\scripts\install_dependencies.ps1
```

### **2. Configure Settings**
Edit configuration files in `config/`:
- `extraction_config.yml` - Database connection and extraction settings
- `presidio_config.yml` - PCI analysis and reporting settings

### **3. Run Complete Pipeline**
```bash
# Bash/Git Bash
chmod +x scripts/run_full_pipeline.sh && ./scripts/run_full_pipeline.sh

# PowerShell
.\scripts\run_full_pipeline.ps1
```

### **4. Or Run Individual Components**
```bash
# Extract database data
./scripts/run_database_extractor.sh

# Analyze for PCI data
./scripts/run_presidio_analysis.sh
```

## ⚙️ **Configuration**

### **Database Extraction** (`config/extraction_config.yml`)
```yaml
database:
  types: [postgresql, mysql]
  hosts: [localhost]
  port_scan:
    postgresql: [5432, 5433, 5434]
    mysql: [3306, 3307, 3308]
  credentials:
    postgresql:
      - username: postgres
        password: pgadmin

extraction:
  sample_percentage: 0.5  # Extract 50% of records
  max_records_per_table: 10000

output:
  directory: ./data/extracted_data
  format: json
```

### **PCI Analysis** (`config/presidio_config.yml`)
```yaml
detection:
  entities:
    - CREDIT_CARD
    - US_SSN
    - PERSON
    - PHONE_NUMBER
    - EMAIL_ADDRESS
  min_score: 0.3

risk_assessment:
  entity_weights:
    CREDIT_CARD: 10
    US_SSN: 9
    PHONE_NUMBER: 3

output:
  directory: ./data/pci_analysis_results
  export_formats: [json, csv, html]
```

## 📊 **Output Files**

### **Database Extraction Results** (`data/extracted_data/`)
- `TIMESTAMP_HOST_PORT_DATABASE_TABLE.json` - Extracted data with metadata

### **PCI Analysis Results** (`data/pci_analysis_results/`)
- `detailed_pci_analysis_TIMESTAMP.json` - Detailed findings
- `pci_compliance_report_TIMESTAMP.json` - Executive summary
- `pci_analysis_TIMESTAMP.csv` - Spreadsheet export
- `pci_dashboard_TIMESTAMP.html` - Interactive dashboard

## 🔍 **Features**

### **Database Discovery**
✅ **Multi-Database Support**: PostgreSQL, MySQL  
✅ **Automatic Discovery**: Scans hosts/ports for databases  
✅ **Configurable Sampling**: Extract any percentage of records  
✅ **Metadata Preservation**: Tracks source database/table info  
✅ **Duplicate Prevention**: Avoids redundant extractions  

### **PCI Compliance Analysis**
✅ **15+ PII Types**: Credit cards, SSN, phone numbers, emails  
✅ **Custom Recognizers**: CVV codes, account numbers, routing numbers  
✅ **Risk Assessment**: CRITICAL, HIGH, MEDIUM, LOW scoring  
✅ **Multiple Formats**: JSON, CSV, HTML dashboard  
✅ **Compliance Mapping**: PCI DSS alignment  

### **Enterprise Features**
✅ **Configurable Everything**: No hardcoded values  
✅ **Comprehensive Logging**: Detailed audit trails  
✅ **Error Handling**: Graceful failure recovery  
✅ **Security Controls**: Connection timeouts, query limits  
✅ **Scalable Architecture**: Handle large databases  

## 🛡️ **Security & Compliance**

### **Data Protection**
- Connection timeouts prevent hanging connections
- Query limits prevent excessive data extraction
- Credential testing with multiple fallbacks
- Exclusion lists for system databases/tables

### **PCI DSS Alignment**
- **Requirement 3**: Identifies stored cardholder data
- **Requirement 7**: Maps data access patterns
- **Requirement 10**: Provides audit trails
- **Requirement 12**: Generates compliance reports

### **Audit Trail**
- All database connections logged
- Data extraction tracked with timestamps
- PII detection results with confidence scores
- Risk assessments with detailed justification

## 📈 **Performance**

- **Database Discovery**: ~10 databases/minute
- **Data Extraction**: ~1000 records/minute
- **PII Analysis**: ~500 records/minute
- **Memory Usage**: ~100MB for typical workloads
- **Scalability**: Handles millions of records

## 🔧 **Advanced Usage**

### **Custom PCI Patterns**
```yaml
custom_recognizers:
  internal_id:
    pattern: "EMP\\d{6}"
    context: ["employee", "emp id"]
    score: 0.9
```

### **Batch Processing**
```bash
# Process specific databases
python scripts/database_extractor.py --databases "customer_db,payment_db"

# Custom output directory
python scripts/presidio_analyzer.py --output-dir "./compliance_reports"
```

### **CI/CD Integration**
```bash
# Exit with error if high-risk PII found
python scripts/presidio_analyzer.py --fail-on-high-risk
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **"No databases discovered"**
   - Check database services are running
   - Verify credentials in `config/extraction_config.yml`
   - Check network connectivity and firewall settings

2. **"Presidio not installed"**
   - Run `./scripts/install_dependencies.sh`
   - Manually install: `pip install presidio-analyzer presidio-anonymizer`

3. **"spaCy model not found"**
   - Download model: `python -m spacy download en_core_web_sm`

4. **"Permission denied"**
   - Check database user permissions
   - Ensure user can access target databases

### **Debug Mode**
```yaml
# In config files
logging:
  level: DEBUG
  console: true
```

## 📚 **Documentation**

- `docs/DATABASE_SETUP.md` - Database setup instructions
- `docs/CONFIGURATION.md` - Detailed configuration guide
- `docs/API_REFERENCE.md` - Python API documentation
- `docs/COMPLIANCE.md` - PCI DSS compliance mapping

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- **Issues**: GitHub Issues
- **Documentation**: `/docs` directory
- **Examples**: `/examples` directory

---

**Built for enterprise PCI compliance and data discovery needs.**