import torch
import matplotlib.pyplot as plt


class Spectrum:
    def __init__(self, trained_model_path: str):
        self.trained_model_path = trained_model_path

    def load_model(self):
        checkpoint = torch.load(self.trained_model_path)
        embedding_dim = 768
        loaded_transformation = checkpoint['model_class'](embedding_dim, embedding_dim)
        loaded_transformation.load_state_dict(checkpoint['state_dict'])
        loaded_transformation.eval()
        transformation_matrix = list(loaded_transformation.parameters())[0].detach()
        return transformation_matrix
    
    def compute_eigenvalues(self, matrix):
        return torch.linalg.eigvals(matrix)
    
    def plot_eigenvalues(self, eigenvalues):
        plt.figure(figsize=(8, 8))
        plt.scatter(eigenvalues.real, eigenvalues.imag, color='darkcyan', s=8)
        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        plt.title('Spectrum of the Transformation Matrix for BERT-cls')
        plt.grid(True)
        plt.axhline(0, color='black',linewidth=0.5)
        plt.axvline(0, color='black',linewidth=0.5)
        plt.savefig('./../resources/plots/spectrum/entities_only/spectrum_bert_cls.png')



if __name__ == '__main__':
    
    spectrum = Spectrum('./../../trained_models/entities_only/full_dim/42k_linear_bert_cls.pth')
    matrix = spectrum.load_model()
    eigenvalues = spectrum.compute_eigenvalues(matrix)
    spectrum.plot_eigenvalues(eigenvalues)
