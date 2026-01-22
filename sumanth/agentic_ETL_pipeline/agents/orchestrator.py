"""
Ultra-Simple Orchestrator with 3-Agent Pipeline
"""

import asyncio
import logging
import yaml
import os
from datetime import datetime
from agents.etl_agent import create_etl_agent
from agents.metrics_agent import create_metrics_agent
from agents.neo4j_agent import create_neo4j_agent


def get_timestamp() -> str:
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class SimpleOrchestrator:
    """Minimal orchestrator - 3-agent ETL pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Load config directly
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.etl_agent = create_etl_agent(config_path)
        self.metrics_agent = create_metrics_agent(config_path)
        self.neo4j_agent = create_neo4j_agent(config_path)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format']
        )
        self.logger = logging.getLogger("Orchestrator")
    
    async def process_file(self, file_path: str) -> dict:
        """Process file with ETL → Metrics → Neo4j pipeline"""
        start_time = get_timestamp()
        print(f"[{start_time}] 🚀 Orchestrator: Starting 3-agent pipeline for {file_path}")
        self.logger.info(f"Processing {file_path}")
        
        try:
            timeout = self.config['agent']['timeout_seconds']
            
            # Step 1: ETL Processing
            print(f"[{get_timestamp()}] 📊 Orchestrator: Step 1 - Running ETL Agent...")
            self.logger.info("Step 1: Running ETL Agent...")
            etl_result = await asyncio.wait_for(
                self.etl_agent.process_file(file_path),
                timeout=timeout
            )
            
            if etl_result['status'] != 'success':
                print(f"[{get_timestamp()}] ❌ Orchestrator: ETL Agent failed")
                return etl_result
            
            # Get processed data from ETL agent
            processed_data = self.etl_agent.get_processed_data()
            if processed_data is None:
                print(f"[{get_timestamp()}] ❌ Orchestrator: No processed data available from ETL agent")
                return {'status': 'error', 'error': 'No processed data available from ETL agent'}
            
            # Step 2: Metrics Analysis
            print(f"[{get_timestamp()}] 📈 Orchestrator: Step 2 - Running Metrics Agent...")
            self.logger.info("Step 2: Running Metrics Agent...")
            schema_info = f"Columns: {list(processed_data.columns)}, Shape: {processed_data.shape}"
            
            metrics_result = await asyncio.wait_for(
                self.metrics_agent.analyze_data(processed_data, schema_info),
                timeout=timeout * 2
            )
            
            if metrics_result['status'] != 'success':
                print(f"[{get_timestamp()}] ❌ Orchestrator: Metrics Agent failed")
                return metrics_result
            
            # Step 3: Neo4j Loading
            print(f"[{get_timestamp()}] 🗄️ Orchestrator: Step 3 - Running Neo4j Agent...")
            self.logger.info("Step 3: Running Neo4j Agent...")
            metrics_data = self.metrics_agent.get_metrics_results()
            
            neo4j_result = await asyncio.wait_for(
                self.neo4j_agent.load_to_neo4j(processed_data, metrics_data),
                timeout=timeout * 3  # Neo4j loading might take longer
            )
            
            # Auto-save if configured
            auto_save = self.config.get('file', {}).get('auto_save', False)
            saved_files = []
            
            if auto_save:
                print(f"[{get_timestamp()}] 💾 Orchestrator: Auto-saving outputs...")
                
                # Save processed data
                output_path = self.config.get('file', {}).get('output_path', 'output/processed_data.csv')
                if self.etl_agent.save_processed_data(output_path):
                    saved_files.append(output_path)
                    self.logger.info(f"Saved processed data to {output_path}")
                
                # Save metrics report
                metrics_path = self.config.get('file', {}).get('metrics_path', 'output/metrics_report.json')
                if self.metrics_agent.save_metrics_report(metrics_path):
                    saved_files.append(metrics_path)
                    self.logger.info(f"Saved metrics report to {metrics_path}")
                
                # Save Neo4j load report
                neo4j_report_path = self.config.get('file', {}).get('neo4j_report_path', 'output/neo4j_load_report.json')
                if neo4j_result['status'] == 'success' and self.neo4j_agent.save_load_report(neo4j_report_path):
                    saved_files.append(neo4j_report_path)
                    self.logger.info(f"Saved Neo4j load report to {neo4j_report_path}")
            
            # Combine results
            end_time = get_timestamp()
            combined_result = {
                'status': 'success',
                'pipeline_start': start_time,
                'pipeline_end': end_time,
                'etl_result': etl_result,
                'metrics_result': metrics_result,
                'neo4j_result': neo4j_result,
                'data_shape': processed_data.shape,
                'saved_files': saved_files
            }
            
            print(f"[{end_time}] ✅ Orchestrator: Completed 3-agent pipeline for {file_path}")
            self.logger.info(f"Completed 3-agent pipeline for {file_path}")
            return combined_result
            
        except asyncio.TimeoutError:
            error_result = {'status': 'error', 'error': 'Processing timeout'}
            self.logger.error(f"Timeout processing {file_path}")
            return error_result
            
        except Exception as e:
            error_result = {'status': 'error', 'error': str(e)}
            self.logger.error(f"Failed processing {file_path}: {e}")
            return error_result


async def main():
    """Simple ETL execution flow"""
    print("=== Agentic ETL System ===")
    
    # Initialize orchestrator
    orchestrator = SimpleOrchestrator()
    
    # Check for configured file path
    configured_file = orchestrator.config.get('file', {}).get('input_path', '').strip()
    auto_process = orchestrator.config.get('file', {}).get('auto_process', True)
    
    if configured_file and os.path.exists(configured_file) and auto_process:
        print(f"Found configured file: {configured_file}")
        print("Processing...")
        
        result = await orchestrator.process_file(configured_file)
        
        print(f"Status: {result['status']}")
        if result['status'] == 'success':
            print(f"Result: {result.get('agent_output', 'Processing completed')}")
            if result.get('final_data_shape'):
                print(f"Final shape: {result['final_data_shape']}")
        else:
            print(f"Error: {result.get('error')}")
        
        return
    
    # Interactive mode if no configured file
    print("No file configured. Manual input required.")
    
    while True:
        print("\n1. Upload file")
        print("2. Exit")
        
        choice = input("\nChoice (1-2): ").strip()
        
        if choice == "1":
            file_path = input("Paste file path: ").strip()
            
            if not file_path:
                print("File path required!")
                continue
                
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
            
            print("Processing...")
            result = await orchestrator.process_file(file_path)
            
            print(f"Status: {result['status']}")
            if result['status'] == 'success':
                print(f"Result: {result.get('agent_output', 'Processing completed')}")
                if result.get('final_data_shape'):
                    print(f"Final shape: {result['final_data_shape']}")
            else:
                print(f"Error: {result.get('error')}")
                
        elif choice == "2":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    asyncio.run(main())