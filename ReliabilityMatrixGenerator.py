import numpy as np
from matplotlib import pyplot as plt

class EnhancedPolarCodeDesign:
    def __init__(self, N, K, design_snr=0.0):
        assert (N & (N - 1)) == 0, "N must be a power of 2"
        self.N = N
        self.K = K
        self.design_snr = design_snr
        self.n = int(np.log2(N))
        self.bit_reliability = None
        self.decision_matrix = None
        
    def calculate_reliability(self, method='gaussian'):
        """Calculate reliability metrics with enhanced output"""
        if method == 'bhattacharyya':
            metrics = np.zeros(self.N)
            metrics[0] = 2 * (1 - 10**(self.design_snr/10))
            for level in range(1, self.n + 1):
                s = 2**level
                for j in range(0, s//2):
                    metrics[s//2 + j] = 2 * metrics[j] - metrics[j]**2
                    metrics[j] = metrics[j]**2
            # Convert to probabilities (higher is better)
            reliability = 1 - metrics
            
        elif method == 'gaussian':
            metrics = np.zeros(self.N)
            metrics[0] = 2 * 10**(self.design_snr/10)
            for level in range(1, self.n + 1):
                s = 2**level
                for j in range(0, s//2):
                    metrics[s//2 + j] = phi_inv(1 - (1 - phi(metrics[j]))**2)
                    metrics[j] = 2 * metrics[j]
            reliability = metrics / np.max(metrics)
            
        else:
            raise ValueError("Unknown method")
            
        self.bit_reliability = reliability
        return reliability
    
    def construct_enhanced_matrix(self, method='gaussian'):
        """Create enhanced decision matrix with reliability scores"""
        reliability = self.calculate_reliability(method)
        sorted_indices = np.argsort(-reliability)  # Descending order
        
        # Create enhanced decision matrix
        enhanced_matrix = np.zeros((self.N, 3))
        enhanced_matrix[:, 0] = np.arange(self.N)  # Bit index
        enhanced_matrix[:, 1] = reliability  # Reliability score
        enhanced_matrix[sorted_indices[:self.K], 2] = 1  # Decision (1=info bit)
        
        # Sort by bit index for final output
        self.decision_matrix = enhanced_matrix[enhanced_matrix[:, 0].argsort()]
        return self.decision_matrix
    
    def visualize_reliability(self):
        """Plot reliability of all bit channels"""
        if self.bit_reliability is None:
            self.calculate_reliability()
            
        plt.figure(figsize=(10, 5))
        plt.bar(range(self.N), self.bit_reliability)
        plt.xlabel('Bit Channel Index')
        plt.ylabel('Normalized Reliability')
        plt.title('Polar Code Bit Channel Reliability')
        plt.axhline(y=np.sort(self.bit_reliability)[-self.K], color='r', linestyle='--')
        plt.grid(True)
        plt.show()
    
    def get_detailed_info(self):
        """Return detailed information about bit channels"""
        info_bits = self.decision_matrix[self.decision_matrix[:, 2] == 1]
        frozen_bits = self.decision_matrix[self.decision_matrix[:, 2] == 0]
        
        return {
            'information_bits': info_bits,
            'frozen_bits': frozen_bits,
            'reliability_threshold': np.sort(self.bit_reliability)[-self.K],
            'worst_information_bit': np.min(info_bits[:, 1]),
            'best_frozen_bit': np.max(frozen_bits[:, 1])
        }

# Helper functions (same as before)
def phi(x):
    if x < 0.000001: return 1.0
    if x > 50: return 0.0
    return np.exp(-0.4527 * x**0.86 + 0.0218)

def phi_inv(y):
    if y > 1.0: return 0.0
    if y < 0.000001: return 50.0
    if y > 0.999999: return 0.0
    return ((0.0218 - np.log(y)) / 0.4527)**(1/0.86)

# Example usage
if __name__ == "__main__":
    # Initialize with larger code size
    pc = EnhancedPolarCodeDesign(N=16, K=8, design_snr=1.0)
    
    # Construct enhanced decision matrix
    enhanced_matrix = pc.construct_enhanced_matrix(method='gaussian')
    
    print("Enhanced Decision Matrix:")
    print("Columns: [Bit Index, Reliability Score, Decision (1=info)]")
    print(enhanced_matrix)
    
    # Visualize reliability
    pc.visualize_reliability()
    
    # Get detailed analysis
    details = pc.get_detailed_info()
    print("\nDesign Analysis:")
    print(f"Reliability threshold for info bits: {details['reliability_threshold']:.4f}")
    print(f"Worst information bit reliability: {details['worst_information_bit']:.4f}")
    print(f"Best frozen bit reliability: {details['best_frozen_bit']:.4f}")