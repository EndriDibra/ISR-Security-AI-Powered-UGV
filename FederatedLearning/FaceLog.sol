// SPDX-License-Identifier: MIT 
pragma solidity ^0.8.20;

/**
 * @title FaceLog
 * @dev A smart contract to log face recognition events and their categories.
 * This contract is designed for a hybrid AI/blockchain system where a face
 * recognition model logs events to an immutable public record.
 */

contract FaceLog {
    
    // Structure to hold a single face recognition event
    struct LogEvent {
        uint256 timestamp;
        string faceLabel;
        uint256 confidence; // Stored as a percentage (e.g., 95 for 95.00%)
        string category;
    }

    // A dynamic array to store all logged events
    LogEvent[] public logEvents;

    // Mapping to track if a face has been logged for a specific address
    mapping(address => bool) public hasFace;

    // Mapping to store the last logged face hash for an address
    mapping(address => bytes32) public faceHashes;

    // Event to emit whenever a new face recognition log is added
    event FaceLogged(
        address indexed user,
        uint256 timestamp,
        string faceLabel,
        uint256 confidence,
        string category
    );

    // Contract owner
    address private owner;

    constructor() {
        owner = msg.sender;
    }

    // Modifier to ensure only owner can call restricted functions
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function.");
        _;
    }

    /**
     * @dev Logs a face recognition event to the blockchain.
     * @param _faceLabel A string representing the identified face (e.g., "bezos", "unknown").
     * @param _confidence The confidence score of the recognition, as a percentage integer (0-100).
     * @param _category The category of the face, e.g., "Authorized" or "Unauthorized".
     */
    function logEvent(
        string calldata _faceLabel,
        uint256 _confidence,
        string calldata _category
    ) public {

        logEvents.push(
            LogEvent({
                timestamp: block.timestamp,
                faceLabel: _faceLabel,
                confidence: _confidence,
                category: _category
            })
        );

        hasFace[msg.sender] = true;
        faceHashes[msg.sender] = keccak256(
            abi.encodePacked(_faceLabel, _confidence, _category, block.timestamp)
        );

        emit FaceLogged(msg.sender, block.timestamp, _faceLabel, _confidence, _category);
    }

    /**
     * @dev Retrieves a single log event by its index.
     * @param _index The index of the event in the logEvents array.
     * @return timestamp The timestamp of the event.
     * @return faceLabel The label of the face.
     * @return confidence The confidence score.
     * @return category The category of the face.
     */
    function getLog(uint256 _index) public view returns (
        uint256 timestamp,
        string memory faceLabel,
        uint256 confidence,
        string memory category
    ) {
        require(_index < logEvents.length, "Log index out of bounds.");
        LogEvent storage eventLog = logEvents[_index];
        return (eventLog.timestamp, eventLog.faceLabel, eventLog.confidence, eventLog.category);
    }

    /**
     * @dev Returns the total number of logged events.
     * @return The number of logs.
     */
    function getLogCount() public view returns (uint256) {
        return logEvents.length;
    }
}
