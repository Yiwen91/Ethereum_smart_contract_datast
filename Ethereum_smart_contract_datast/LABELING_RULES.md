# Vulnerability Labeling Rules

This document describes the exact labeling rules applied to functions extracted from Solidity contracts.

## SWC ID Mapping

| Messi-Q Vulnerability | SWC ID | SWC Name (SmartBugs-Wild) |
|------------------------|--------|---------------------------|
| Reentrancy | SWC-107 | Reentrancy |
| Timestamp Dependency | SWC-116 | Block Timestamp Dependence |
| Integer Overflow/Underflow | SWC-101 | Integer Overflow and Underflow |
| Dangerous Delegatecall | SWC-112 | Delegatecall to Untrusted Contract |
| Transaction-Ordering Dependence | SWC-114 | Transaction-Ordering Dependence |
| Uninitialized Storage Pointer | SWC-109 | Uninitialized Storage Pointer |
| Unchecked External Calls | SWC-104 | Unchecked Call Return Value |

## Labeling Rules

### 1. Timestamp Dependency (SWC-116)

**Rule**: `TimestampInvoc ∧ (TimestampAssign ∨ TimestampContaminate)`

A function is labeled as having **Timestamp Dependency** if:
- It invokes timestamp-related variables (`TimestampInvoc` = true), AND
- It either assigns timestamp values (`TimestampAssign` = true) OR uses timestamps in conditional operations (`TimestampContaminate` = true)

#### Components:

- **TimestampInvoc**: 
  - Detects: `now`, `block.timestamp`, `block.number`
  - Pattern: Function contains any of these keywords

- **TimestampAssign**:
  - Detects: Assignment of timestamp values
  - Pattern: `now =`, `block.timestamp =`, `block.number =`

- **TimestampContaminate**:
  - Detects: Timestamp used in conditional operations or logic
  - Pattern: Timestamp appears in context with operators (`=`, `<`, `>`, `!`, `&`, `|`)
  - Example: `if (now > deadline)`, `require(block.timestamp < endTime)`

#### Example:

```solidity
function gameResult() private {
    uint index = getRamdon();
    address lastAddress = addressArray[index];
    uint totalBalace = address(this).balance;
    owner.transfer(giveToOwn);
    emit gameOverEvent(gameIndex, curConfig.totalSize, curConfig.singlePrice, 
                      curConfig.pumpRate, lastAddress, now);  // TimestampInvoc
}
```

**Label**: `TimestampInvoc = true`, `TimestampAssign = false`, `TimestampContaminate = false`
**Vulnerability**: Not labeled (missing TimestampAssign or TimestampContaminate)

```solidity
function getRamdon() private view returns (uint) {
    bytes32 ramdon = keccak256(abi.encodePacked(ramdon, now, blockhash(block.number-1)));  // TimestampInvoc
    for(uint i = 0; i < addressArray.length; i++) {
        ramdon = keccak256(abi.encodePacked(ramdon, now, addressArray[i]));  // TimestampInvoc
    }
    uint index = uint(ramdon) % addressArray.length;
    return index;
}
```

**Label**: `TimestampInvoc = true`, `TimestampContaminate = true` (used in operations)
**Vulnerability**: **Timestamp Dependency** (SWC-116)

### 2. Reentrancy (SWC-107)

**Rule**: Function contains external calls that may allow reentrancy, **unless** it has a ReentrancyGuard/nonReentrant modifier (common fix).

A function is labeled as having **Reentrancy** if:
- It contains external calls: `.call()`, `.send()`, `.transfer()`, `.delegatecall()`, `.callcode()`
- **And** it does **not** have `nonReentrant`, `ReentrancyGuard`, or `reentrancyGuard` modifier

**Over-labeling fix**: If the function has a reentrancy guard modifier, it is **not** labeled as reentrant (avoids ~66%+ false positives in datasets).

#### Patterns Detected:
- `.call(`
- `.send(`
- `.transfer(`
- `.delegatecall(`
- `.callcode(`

#### Example:

```solidity
function stopGame() onlyOwner private {
    if(addressArray.length <= 0) {
        return;
    }
    uint totalBalace = address(this).balance;
    uint price = totalBalace / addressArray.length;
    for(uint i = 0; i < addressArray.length; i++) {
        address curPlayer = addressArray[i];
        curPlayer.transfer(price);  // External call - potential reentrancy
    }
    emit stopGameEvent(totalBalace, addressArray.length, price);
    addressArray.length = 0;  // State change after external call
}
```

**Label**: `Reentrancy = true`
**Vulnerability**: **Reentrancy** (SWC-107)

### 3. Integer Overflow/Underflow (SWC-101)

**Rule**: Function contains arithmetic operations that may cause overflow/underflow, **only for Solidity &lt; 0.8**.

A function is labeled as having **Integer Overflow/Underflow** if:
- It contains arithmetic operations: `++`, `--`, `+=`, `-=`, `*=`, `/=`
- It performs arithmetic on integer types (`uint`, `int`)
- **And** the contract uses Solidity **&lt; 0.8.0** (parsed from `pragma solidity`)

**False-positive fix**: Solidity 0.8.0+ has native overflow/underflow protection. Integer overflow labeling is **skipped** for all contracts with `pragma solidity ^0.8.0` or higher.

#### Patterns Detected:
- `++`, `--`
- `+=`, `-=`, `*=`, `/=`
- Arithmetic operations on `uint`/`int` types

#### Example:

```solidity
function startNewGame() private {
    gameIndex++;  // Potential overflow
    if(curConfig.hasChange) {
        if(curConfig.totalSize != setConfig.totalSize) {
            curConfig.totalSize = setConfig.totalSize;
        }
    }
    addressArray.length = 0;
}
```

**Label**: `IntegerOverflow = true`
**Vulnerability**: **Integer Overflow/Underflow** (SWC-101)

### 4. Dangerous Delegatecall (SWC-112)

**Rule**: Function uses `delegatecall` which can be dangerous with untrusted contracts

A function is labeled as having **Dangerous Delegatecall** if:
- It contains `.delegatecall(` or `delegatecall(`

#### Patterns Detected:
- `.delegatecall(`
- `delegatecall(`

#### Example:

```solidity
function executeDelegatecall(address target, bytes memory data) public {
    (bool success, ) = target.delegatecall(data);  // Dangerous delegatecall
    require(success, "Delegatecall failed");
}
```

**Label**: `Delegatecall = true`
**Vulnerability**: **Dangerous Delegatecall** (SWC-112)

### 5. Transaction-Ordering Dependence (SWC-114)

**Rule**: Public/external function manipulates shared order-sensitive state and uses competitive logic or value flow.

A function is labeled as having **Transaction-Ordering Dependence** if:
- It is `public` or `external`
- It reads or writes shared order-sensitive variables such as `highestBid`, `price`, `order`, `orders`, `winner`, `pending`, `balances`, `allowance`
- It also contains comparisons (`>`, `<`, `>=`, `<=`, `==`, `!=`) or value flow such as `msg.value`, `transfer`, `send`, `call`, `approve`, `transferFrom`

#### Example:

```solidity
function bid() public payable {
    require(msg.value > highestBid);
    highestBid = msg.value;
    winner = msg.sender;
}
```

**Label**: `TransactionOrderingDependence = true`
**Vulnerability**: **Transaction-Ordering Dependence** (SWC-114)

### 6. Uninitialized Storage Pointer (SWC-109)

**Rule**: Function declares a local `storage` pointer without initialization.

A function is labeled as having **Uninitialized Storage Pointer** if:
- It contains a local declaration like `Type storage name;`
- The declaration appears inside the function body, not in the function signature

#### Example:

```solidity
function update() public {
    Data storage d;
    d.x = 1;
}
```

**Label**: `UninitializedStoragePointer = true`
**Vulnerability**: **Uninitialized Storage Pointer** (SWC-109)

### 7. Unchecked External Calls (SWC-104)

**Rule**: Function makes a low-level external call and ignores the boolean result.

A function is labeled as having **Unchecked External Calls** if:
- It contains `.call(`, `.send(`, `.delegatecall(`, or `.callcode(`
- The result is not wrapped directly in `require`, `assert`, or `if`
- The result is not assigned to a variable on that line

#### Example:

```solidity
function ping(address target) public {
    target.call("hello");
}
```

**Label**: `UncheckedExternalCalls = true`
**Vulnerability**: **Unchecked External Calls** (SWC-104)

## Output Format

### Function Data Structure

Each extracted function includes:

```python
{
    "contract_file": "path/to/contract.sol",
    "contract_name": "MyContract",
    "function_name": "vulnerableFunction",
    "function_signature": "vulnerableFunction(uint256)",
    "function_code": "function vulnerableFunction(uint256 x) public {...}",
    "start_line": 10,
    "end_line": 25,
    "visibility": "public",
    "state_mutability": "",
    "vulnerabilities": ["Timestamp Dependency", "Reentrancy"],
    "swc_ids": ["SWC-116", "SWC-107"],
    "labels": {
        "TimestampInvoc": true,
        "TimestampAssign": false,
        "TimestampContaminate": true,
        "Reentrancy": true,
        "IntegerOverflow": false,
        "Delegatecall": false,
        "TransactionOrderingDependence": false,
        "UninitializedStoragePointer": false,
        "UncheckedExternalCalls": false
    },
    "metadata": {}
}
```

## Implementation Notes

1. **Pattern Matching**: The current implementation uses regex-based pattern matching. For production use, consider integrating with more sophisticated static analysis tools (e.g., Slither's built-in detectors).

2. **False Positives**: Heuristic-based detection may produce false positives. Manual review is recommended for critical applications.

3. **False Negatives**: Some vulnerabilities may not be detected due to complex control flow or indirect patterns. Consider using multiple analysis tools.

4. **Timestamp Contamination**: The detection of `TimestampContaminate` uses context analysis (50 characters before/after) to identify conditional usage. This is a simplified approach.

5. **Reentrancy Detection**: Full reentrancy detection requires control flow analysis to determine if state changes occur before external calls. The current implementation is a basic heuristic.

## Extending Labeling Rules

To add new vulnerability types:

1. Add detection method in `VulnerabilityLabeler` class
2. Update `label_function` method
3. Add SWC mapping in `swc_mapping.json`
4. Update CSV export fieldnames if needed
