"""
CPE Enricher for ORBIT Node Pipeline
Enriches NJS_Software nodes with CPE strings for accurate vulnerability linking

This script acts as a standalone CPE enrichment layer that:
1. Normalizes vendor and product names
2. Constructs CPE 2.3 formatted strings
3. Validates CPEs against Core graph and NVD
4. Outputs enriched software data with CPE metadata

TODO: Integrate as Step 2.5 in pipeline (between extractors and graph builder)
"""

import os
import re
import json
import yaml
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Third-party imports
try:
    import nvdlib
    from neo4j import GraphDatabase
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install nvdlib neo4j python-dotenv")
    exit(1)

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%dT%H:%M:%SZ'
)
logger = logging.getLogger('cpe_enricher')


class CPEEnricher:
    """CPE enrichment service for software packages"""
    
    def __init__(self, vendor_map_path: str = "cpe_builder/vendor_map.yaml"):
        """Initialize CPE enricher"""
        self.vendor_map = self._load_vendor_map(vendor_map_path)
        self.cpe_config = self._load_cpe_config()
        self.neo4j_driver = self._init_neo4j_connection()
        self.nvd_api_key = os.getenv('NVD_API_KEY', '')
        self.stats = {
            'total': 0,
            'matched_core': 0,
            'matched_nvd': 0,
            'unverified': 0,
            'errors': 0
        }
        logger.info("CPE Enricher initialized")
    
    def _load_cpe_config(self) -> Dict[str, Any]:
        """Load CPE enrichment configuration"""
        try:
            config_file = Path("config/cpe_enrichment_config.yaml")
            if not config_file.exists():
                script_dir = Path(__file__).parent.parent
                config_file = script_dir / "config/cpe_enrichment_config.yaml"
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded CPE enrichment config from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load CPE enrichment config: {e}")
            return {}
    
    def _load_vendor_map(self, path: str) -> Dict[str, str]:
        """Load vendor normalization map from YAML"""
        try:
            vendor_file = Path(path)
            if not vendor_file.exists():
                # Try relative to script location
                script_dir = Path(__file__).parent
                vendor_file = script_dir / "vendor_map.yaml"
            
            with open(vendor_file, 'r') as f:
                config = yaml.safe_load(f)
            
            vendor_map = config.get('vendor_mappings', {})
            logger.info(f"Loaded {len(vendor_map)} vendor mappings from {vendor_file}")
            return vendor_map
        
        except Exception as e:
            logger.error(f"Failed to load vendor map: {e}")
            return {}
    
    def _init_neo4j_connection(self) -> Optional[GraphDatabase.driver]:
        """Initialize Neo4j connection to Core graph"""
        try:
            uri = os.getenv('NEO4J_URI')
            username = os.getenv('NEO4J_USERNAME')
            password = os.getenv('NEO4J_PASSWORD')
            
            if not all([uri, username, password]):
                logger.warning("Neo4j credentials not found in environment - Core graph matching disabled")
                return None
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            with driver.session() as session:
                session.run("RETURN 1")
            
            logger.info(f"Connected to Neo4j Core graph at {uri}")
            return driver
        
        except Exception as e:
            logger.warning(f"Failed to connect to Neo4j Core graph: {e}")
            logger.warning("Falling back to NVD-only mode")
            return None
    
    def normalize_vendor(self, vendor: str) -> str:
        """
        Normalize vendor name to NVD-compatible format
        
        Args:
            vendor: Raw vendor string from Wazuh
        
        Returns:
            Normalized vendor name (lowercase, no special chars)
        """
        if not vendor:
            return "unknown"
        
        # Check vendor map first
        if vendor in self.vendor_map:
            return self.vendor_map[vendor]
        
        # Fallback: basic normalization
        normalized = vendor.lower()
        normalized = re.sub(r'[,\.]', '', normalized)  # Remove commas and periods
        normalized = re.sub(r'\s+', '_', normalized)   # Replace spaces with underscores
        normalized = re.sub(r'[^a-z0-9_-]', '', normalized)  # Remove special chars
        
        logger.debug(f"Vendor '{vendor}' not in map, normalized to '{normalized}'")
        return normalized
    
    def normalize_product(self, name: str, vendor: str = "") -> str:
        """
        Normalize product name to NVD-compatible format
        
        Pure function for easy unit testing
        
        Args:
            name: Raw product name from Wazuh
            vendor: Vendor name (to strip from product if present)
        
        Returns:
            Normalized product name
        """
        if not name:
            return "unknown"
        
        product = name.lower()
        
        # Remove vendor name if it appears in product
        if vendor:
            vendor_lower = vendor.lower()
            product = re.sub(rf'\b{re.escape(vendor_lower)}\b', '', product, flags=re.IGNORECASE)
        
        # Remove architecture suffixes
        product = re.sub(r'\s*\(x64\)', '', product, flags=re.IGNORECASE)
        product = re.sub(r'\s*\(x86\)', '', product, flags=re.IGNORECASE)
        product = re.sub(r'\s*\(64-bit\)', '', product, flags=re.IGNORECASE)
        product = re.sub(r'\s*\(32-bit\)', '', product, flags=re.IGNORECASE)
        
        # Remove "(User)" suffix
        product = re.sub(r'\s*\(user\)', '', product, flags=re.IGNORECASE)
        
        # Remove version numbers embedded in name (e.g., "Visual C++ 2022")
        # Keep version if it's part of the product identity
        product = re.sub(r'\s+\d+\.\d+\.\d+.*$', '', product)  # Remove trailing versions
        
        # Remove "minimum runtime", "redistributable" etc
        product = re.sub(r'\s+(minimum|maximum|redistributable|runtime|additional|installer).*$', '', product, flags=re.IGNORECASE)
        
        # Clean up whitespace
        product = re.sub(r'\s+', ' ', product).strip()
        
        # Replace spaces with underscores
        product = product.replace(' ', '_')
        
        # Remove special characters except underscores, hyphens, dots, and plus
        product = re.sub(r'[^a-z0-9_\-\.\+]', '', product)
        
        # Remove leading/trailing underscores
        product = product.strip('_')
        
        return product if product else "unknown"
    
    def construct_cpe(self, vendor: str, product: str, version: str) -> str:
        """
        Construct CPE 2.3 formatted string
        
        Format: cpe:2.3:a:{vendor}:{product}:{version}:*:*:*:*:*:*:*
        
        Args:
            vendor: Normalized vendor name
            product: Normalized product name
            version: Software version
        
        Returns:
            CPE 2.3 formatted string
        """
        # Ensure version is clean
        version = version.strip() if version else "*"
        
        cpe = f"cpe:2.3:a:{vendor}:{product}:{version}:*:*:*:*:*:*:*"
        return cpe
    
    def match_core_graph(self, product: str, version: str) -> Dict[str, Any]:
        """
        Layer 1: Match against Core graph CVEs
        
        Args:
            product: Normalized product name
            version: Software version
        
        Returns:
            Match result with status and matched CPEs
        """
        if not self.neo4j_driver:
            return {'status': 'CORE_UNAVAILABLE', 'matched_cpes': [], 'matched_cve_ids': []}
        
        try:
            with self.neo4j_driver.session() as session:
                # Get query from config
                queries = self.cpe_config.get('queries', {})
                query = queries.get('match_core_cves', 'MATCH (c:NJS_CVE) WHERE any(cpe IN c.cpes WHERE toLower(cpe) CONTAINS toLower($product)) RETURN c.id as cve_id, c.cpes as cpes LIMIT 10')
                
                result = session.run(query, product=product)
                records = list(result)
                
                if not records:
                    logger.debug(f"No Core graph match for product: {product}")
                    return {'status': 'NO_MATCH', 'matched_cpes': [], 'matched_cve_ids': []}
                
                # Extract matching CPEs and CVE IDs
                matched_cpes = []
                matched_cve_ids = []
                
                for record in records:
                    cve_id = record['cve_id']
                    cpes = record['cpes']
                    
                    matched_cve_ids.append(cve_id)
                    
                    # Find CPEs that match both product and version
                    for cpe in cpes:
                        cpe_lower = cpe.lower()
                        if product.lower() in cpe_lower:
                            # Check if version matches (optional - can be fuzzy)
                            if version and version.lower() in cpe_lower:
                                matched_cpes.append(cpe)
                            elif not version:
                                matched_cpes.append(cpe)
                
                if matched_cpes:
                    logger.info(f"Core graph match found: {len(matched_cpes)} CPEs, {len(matched_cve_ids)} CVEs")
                    return {
                        'status': 'MATCHED_CORE',
                        'matched_cpes': list(set(matched_cpes)),  # Remove duplicates
                        'matched_cve_ids': matched_cve_ids
                    }
                else:
                    return {'status': 'NO_MATCH', 'matched_cpes': [], 'matched_cve_ids': matched_cve_ids}
        
        except Exception as e:
            logger.error(f"Core graph query failed: {e}")
            return {'status': 'CORE_ERROR', 'matched_cpes': [], 'matched_cve_ids': []}
    
    def match_nvd(self, name: str, version: str) -> Dict[str, Any]:
        """
        Layer 2: Match against NVD API (fallback)
        
        Args:
            name: Original product name
            version: Software version
        
        Returns:
            Match result with status and matched CPE
        """
        try:
            # Respect NVD rate limit: 5 requests per 30 seconds (without API key)
            time.sleep(6)  # 6 seconds between requests
            
            keyword = f"{name} {version}" if version else name
            logger.debug(f"Searching NVD for: {keyword}")
            
            # Search NVD CPE database
            cpes = nvdlib.searchCPE(keywordSearch=keyword)
            
            if cpes:
                # Get first matching CPE
                first_cpe = cpes[0]
                cpe_string = first_cpe.cpeName if hasattr(first_cpe, 'cpeName') else str(first_cpe)
                
                logger.info(f"NVD match found: {cpe_string}")
                return {
                    'status': 'MATCHED_NVD',
                    'matched_cpe': cpe_string
                }
            else:
                logger.debug(f"No NVD match for: {keyword}")
                return {'status': 'NO_MATCH', 'matched_cpe': None}
        
        except Exception as e:
            logger.error(f"NVD API query failed: {e}")
            return {'status': 'NVD_ERROR', 'matched_cpe': None}
    
    def enrich_software(self, software: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single software package with CPE data
        
        Args:
            software: Software dictionary with name, version, vendor
        
        Returns:
            Enriched software dictionary with CPE metadata
        """
        name = software.get('name', '')
        version = software.get('version', '')
        vendor = software.get('vendor', '')
        
        logger.info(f"Enriching: {name} {version} ({vendor})")
        
        # Step 1: Normalize vendor and product
        normalized_vendor = self.normalize_vendor(vendor)
        normalized_product = self.normalize_product(name, vendor)
        
        # Step 2: Construct CPE
        constructed_cpe = self.construct_cpe(normalized_vendor, normalized_product, version)
        
        # Step 3: Two-layer matching
        # Layer 1: Core graph
        core_match = self.match_core_graph(normalized_product, version)
        
        if core_match['status'] == 'MATCHED_CORE':
            # Core match found - use it
            matched_cpe = core_match['matched_cpes'][0] if core_match['matched_cpes'] else constructed_cpe
            match_status = 'MATCHED_CORE'
            matched_cve_ids = core_match['matched_cve_ids']
            self.stats['matched_core'] += 1
        
        else:
            # Layer 2: NVD fallback
            nvd_match = self.match_nvd(name, version)
            
            if nvd_match['status'] == 'MATCHED_NVD':
                matched_cpe = nvd_match['matched_cpe']
                match_status = 'MATCHED_NVD'
                matched_cve_ids = []
                self.stats['matched_nvd'] += 1
            else:
                # No match - use constructed CPE
                matched_cpe = constructed_cpe
                match_status = 'UNVERIFIED'
                matched_cve_ids = []
                self.stats['unverified'] += 1
        
        # Build enriched result
        enriched = {
            **software,  # Keep original fields
            'normalized_vendor': normalized_vendor,
            'normalized_product': normalized_product,
            'constructed_cpe': constructed_cpe,
            'matched_cpe': matched_cpe,
            'match_status': match_status,
            'matched_cve_ids': matched_cve_ids,
            'enriched_at': datetime.utcnow().isoformat() + 'Z'
        }
        
        logger.info(f"Enriched: {name} → {match_status} → {matched_cpe}")
        return enriched
    
    def enrich_batch(self, software_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich a batch of software packages
        
        Args:
            software_list: List of software dictionaries
        
        Returns:
            List of enriched software dictionaries
        """
        start_time = time.time()
        self.stats['total'] = len(software_list)
        
        logger.info(f"Starting CPE enrichment for {len(software_list)} packages")
        
        enriched_list = []
        
        for idx, software in enumerate(software_list, 1):
            try:
                logger.info(f"Processing {idx}/{len(software_list)}")
                enriched = self.enrich_software(software)
                enriched_list.append(enriched)
            
            except Exception as e:
                logger.error(f"Failed to enrich {software.get('name', 'unknown')}: {e}")
                self.stats['errors'] += 1
                # Add original software with error status
                enriched_list.append({
                    **software,
                    'match_status': 'ERROR',
                    'error': str(e),
                    'enriched_at': datetime.utcnow().isoformat() + 'Z'
                })
        
        duration = time.time() - start_time
        
        # Log statistics
        logger.info("=" * 60)
        logger.info("CPE Enrichment Complete")
        logger.info("=" * 60)
        logger.info(f"Total packages: {self.stats['total']}")
        logger.info(f"Matched (Core): {self.stats['matched_core']}")
        logger.info(f"Matched (NVD):  {self.stats['matched_nvd']}")
        logger.info(f"Unverified:     {self.stats['unverified']}")
        logger.info(f"Errors:         {self.stats['errors']}")
        logger.info(f"Duration:       {duration:.2f} seconds")
        logger.info("=" * 60)
        
        return enriched_list
    
    def save_results(self, enriched_list: List[Dict[str, Any]], output_dir: str = "cpe_builder/output"):
        """
        Save enriched results to JSON file
        
        Args:
            enriched_list: List of enriched software dictionaries
            output_dir: Output directory path
        """
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"cpe_enrichment_results_{timestamp}.json"
            filepath = output_path / filename
            
            # Save to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(enriched_list, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {filepath}")
            logger.info(f"Total records: {len(enriched_list)}")
        
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def close(self):
        """Close Neo4j connection"""
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info("Neo4j connection closed")


def main():
    """Main entry point for standalone execution"""
    logger.info("=" * 60)
    logger.info("CPE Enricher - Standalone Mode")
    logger.info("=" * 60)
    
    # TODO: In pipeline integration, this will be called from Step 2.5
    # Input will come from extractor output (shared_data/data/extracted/{timestamp}/nodes/software_nodes.json)
    # Output will feed into graph builder (3_graph)
    
    # Get shared data path from environment
    shared_data_path = os.getenv('SHARED_DATA_HOST_PATH', '../njs_shared_data')
    extracted_data_dir = Path(shared_data_path) / 'data' / 'extracted'
    
    logger.info(f"Looking for extracted data in: {extracted_data_dir}")
    
    # Find latest extraction session
    software_file = None
    if extracted_data_dir.exists():
        # Find latest timestamped folder
        timestamped_dirs = [d for d in extracted_data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
        if timestamped_dirs:
            latest_dir = max(timestamped_dirs, key=lambda d: d.name)
            software_file = latest_dir / 'nodes' / 'software_nodes.json'
            
            if not software_file.exists():
                # Try without nodes subdirectory
                software_file = latest_dir / 'software_nodes.json'
            
            if software_file.exists():
                logger.info(f"Found software data: {software_file}")
            else:
                logger.warning(f"No software_nodes.json found in {latest_dir}")
                software_file = None
    
    # Load software data
    if software_file and software_file.exists():
        logger.info(f"Loading software data from: {software_file}")
        with open(software_file, 'r', encoding='utf-8') as f:
            software_list = json.load(f)
        logger.info(f"Loaded {len(software_list)} software packages")
    else:
        logger.warning("No extracted software data found, using sample data")
        # Fallback to sample data for testing
        software_list = [
            {
                "name": "Google Chrome",
                "version": "145.0.7632.116",
                "vendor": "Google LLC",
                "agent_id": "001"
            },
            {
                "name": "Microsoft Edge",
                "version": "145.0.3800.70",
                "vendor": "Microsoft Corporation",
                "agent_id": "001"
            },
            {
                "name": "Notepad++ (64-bit x64)",
                "version": "8.7.4",
                "vendor": "Notepad++ Team",
                "agent_id": "001"
            }
        ]
        logger.info(f"Using {len(software_list)} sample packages")
    
    # Initialize enricher
    enricher = CPEEnricher()
    
    try:
        # Enrich software packages
        enriched_results = enricher.enrich_batch(software_list)
        
        # Save results
        enricher.save_results(enriched_results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("Enrichment Summary:")
        print("=" * 60)
        for software in enriched_results[:10]:  # Show first 10
            print(f"\nName: {software['name']}")
            print(f"Status: {software['match_status']}")
            print(f"CPE: {software['matched_cpe']}")
            if software.get('matched_cve_ids'):
                print(f"CVEs: {len(software['matched_cve_ids'])} found")
        
        if len(enriched_results) > 10:
            print(f"\n... and {len(enriched_results) - 10} more packages")
    
    finally:
        enricher.close()


if __name__ == "__main__":
    main()
