import json 
import os
import time
from web3 import Web3
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Connect to Ganache
ganache_url = "http://127.0.0.1:7545"
w3 = Web3(Web3.HTTPProvider(ganache_url))

# Check if connected
if not w3.is_connected():
    print("Error: Could not connect to Ganache. Please ensure it is running.")
    exit()

print(f"Connected to Ganache version: {w3.client_version}")

# Load the deployed contract address and ABI
try:
    with open("contract_details.json", "r") as file:
        contract_data = json.load(file)

    contract_address = contract_data["address"]
    contract_abi = contract_data["abi"]

except FileNotFoundError:
    print("Error: contract_details.json not found. Please run deployContract.py first.")
    exit()

# Create a contract instance
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Get the private key from the environment variable
private_key = os.getenv("PRIVATE_KEY")
if not private_key:
    print("Error: PRIVATE_KEY environment variable not set.")
    exit()

# The account derived from the private key
account = w3.eth.account.from_key(private_key).address


# Function to determine the category based on face label
def get_category(face_label):
    """Determines if a user is authorized or unauthorized."""
    authorized_users = ["bezos", "zuckerberg"]

    if face_label.lower() in authorized_users:
        return "Authorized"
    else:
        return "Unauthorized"


def log_to_blockchain(face_label, confidence, category):
    """
    Logs a face recognition event to the blockchain.
    """
    try:
        nonce = w3.eth.get_transaction_count(account, "pending")

        transaction = contract.functions.logEvent(
            face_label,
            int(confidence * 100),  # Store as integer percentage (95 for 95%)
            category
        ).build_transaction({
            "gasPrice": w3.eth.gas_price,
            "chainId": w3.eth.chain_id,
            "from": account,
            "nonce": nonce,
            "gas": 200000
        })
        
        signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        
        print(f"Transaction sent. Waiting for receipt... Hash: {w3.to_hex(tx_hash)}")
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        print("Transaction successful!")
        print(f"Event logged on block: {tx_receipt.blockNumber}")

    except Exception as e:
        print(f"An error occurred while logging to the blockchain: {e}")


if __name__ == "__main__":
    # Simulate data from your hybrid model
    sample_data = [
        {"label": "bezos", "confidence": 0.95},
        {"label": "unknown", "confidence": 0.78},
        {"label": "zuckerberg", "confidence": 0.89},
        {"label": "elon_musk", "confidence": 0.65},
    ]

    print("Starting blockchain logging simulation...")
    
    for event in sample_data:
        label = event["label"]
        confidence = event["confidence"]
        category = get_category(label)
        
        print(f"\nProcessing event: Label={label}, Confidence={confidence}, Category={category}")
        log_to_blockchain(label, confidence, category)
        time.sleep(1)  # Delay to avoid nonce issues
    
    print("\nSimulation complete.")
    
    # Verify logs
    try:
        log_count = contract.functions.getLogCount().call()
        print(f"\nTotal logs on the blockchain: {log_count}")

        if log_count > 0:
            last_log = contract.functions.getLog(log_count - 1).call()
            print(f"Last logged event details:")
            print(f"  Timestamp: {last_log[0]}")
            print(f"  Face Label: {last_log[1]}")
            print(f"  Confidence: {last_log[2]}%")
            print(f"  Category: {last_log[3]}")

    except Exception as e:
        print(f"Could not retrieve logs from blockchain: {e}")
