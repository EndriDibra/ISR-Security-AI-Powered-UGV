import os 
import json
from web3 import Web3
from dotenv import load_dotenv
from eth_account import Account
from solcx import compile_standard, install_solc


# Load environment variables from .env file
load_dotenv()

# Ganache RPC endpoint
GANACHE_URL = "http://127.0.0.1:7545"
w3 = Web3(Web3.HTTPProvider(GANACHE_URL))

# Check connection
if not w3.is_connected():
    print("Error: Could not connect to Ganache. Please ensure Ganache is running and listening on", GANACHE_URL)
    exit()
else:
    print("Connected to Ganache version:", w3.client_version)

# Check for the private key environment variable
private_key = os.getenv("PRIVATE_KEY")
if not private_key:
    print("Error: PRIVATE_KEY environment variable not set. Please set it with a private key from Ganache.")
    exit()
else:
    print("Private key environment variable found.")

# The account you're deploying from
account = w3.eth.account.from_key(private_key)
print("Deploying from account:", account.address)

# Contract Compilation
try:
    print("Installing Solidity compiler version 0.8.20...")
    install_solc("0.8.20")
    print("Successfully installed solc version 0.8.20.")
except Exception as e:
    print(f"Error installing solc: {e}")
    exit()

contract_name = "FaceLog"
contract_file = f"./{contract_name}.sol"

print(f"Compiling {contract_file} with solc version 0.8.20 and EVM version 'london'...")
with open(contract_file, "r") as file:
    contract_source_code = file.read()

# Compile the contract
try:
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {contract_file: {"content": contract_source_code}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                    }
                },
                "evmVersion": "london"
            },
        },
        solc_version="0.8.20"
    )
except Exception as e:
    print(f"Compilation failed: {e}")
    exit()

# Get bytecode and ABI
bytecode = compiled_sol["contracts"][contract_file][contract_name]["evm"]["bytecode"]["object"]
abi = compiled_sol["contracts"][contract_file][contract_name]["abi"]

# Contract Deployment
Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
nonce = w3.eth.get_transaction_count(account.address, "pending")
chain_id = w3.eth.chain_id

# Build the deployment transaction using constructor
transaction = Contract.constructor().build_transaction({
    'chainId': chain_id,
    'nonce': nonce,
    'gasPrice': w3.eth.gas_price,
    'from': account.address
})

# Estimate gas for the deployment
try:
    gas_estimate = w3.eth.estimate_gas(transaction)
    transaction['gas'] = gas_estimate
    print(f"Gas estimate for deployment: {gas_estimate}")
except Exception as e:
    print(f"Error estimating gas: {e}")
    transaction['gas'] = 2000000 
    print(f"Using fallback gas limit: {transaction['gas']}")

# Sign the transaction
signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)
print("Sending deployment transaction...")

try:
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print("Contract deployed successfully!")
    print("Transaction hash:", tx_receipt.transactionHash.hex())
    print("Contract address:", tx_receipt.contractAddress)

    # Save the contract address and ABI to a file for later use
    with open("contract_details.json", "w") as f:
        json.dump({"address": tx_receipt.contractAddress, "abi": abi}, f, indent=4)

    print("Contract address and ABI saved to contract_details.json")

except Exception as e:
    print(f"An error occurred during transaction sending or waiting for receipt: {e}")
