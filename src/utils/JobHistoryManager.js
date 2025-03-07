// JobHistoryManager.js - Manages estimation history and basket storage

// Storage keys
const HISTORY_STORAGE_KEY = 'fiqci_estimation_history';
const BASKET_STORAGE_KEY = 'fiqci_estimation_basket';

/**
 * Class to manage estimation history and basket
 */
class JobHistoryManager {
	/**
	 * Load history from localStorage
	 * @returns {Array} Array of estimation objects
	 */
	static loadHistory() {
		try {
			const historyJson = localStorage.getItem(HISTORY_STORAGE_KEY);
			return historyJson ? JSON.parse(historyJson) : [];
		} catch (error) {
			console.error('Failed to load history:', error);
			return [];
		}
	}

	/**
	 * Save history to localStorage
	 * @param {Array} history - Array of estimation objects
	 */
	static saveHistory(history) {
		try {
			localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
		} catch (error) {
			console.error('Failed to save history:', error);
		}
	}

	/**
	 * Add a new estimation to history
	 * @param {Object} estimation - Estimation object
	 * @returns {Array} Updated history array
	 */
	static addToHistory(estimation) {
		const history = this.loadHistory();
		const newEstimation = {
			...estimation,
			id: Date.now(), // Use timestamp as unique ID
			timestamp: new Date().toISOString()
		};
		
		const updatedHistory = [newEstimation, ...history];
		this.saveHistory(updatedHistory);
		return updatedHistory;
	}

	/**
	 * Clear the entire history
	 * @returns {Array} Empty array
	 */
	static clearHistory() {
		this.saveHistory([]);
		return [];
	}

	/**
	 * Load basket from localStorage
	 * @returns {Array} Array of estimation objects in basket
	 */
	static loadBasket() {
		try {
			const basketJson = localStorage.getItem(BASKET_STORAGE_KEY);
			return basketJson ? JSON.parse(basketJson) : [];
		} catch (error) {
			console.error('Failed to load basket:', error);
			return [];
		}
	}

	/**
	 * Save basket to localStorage
	 * @param {Array} basket - Array of estimation objects
	 */
	static saveBasket(basket) {
		try {
			localStorage.setItem(BASKET_STORAGE_KEY, JSON.stringify(basket));
		} catch (error) {
			console.error('Failed to save basket:', error);
		}
	}

	/**
	 * Add an estimation to the basket
	 * @param {Object} estimation - Estimation object
	 * @returns {Array} Updated basket array
	 */
	static addToBasket(estimation) {
		const basket = this.loadBasket();
		const updatedBasket = [...basket, estimation];
		this.saveBasket(updatedBasket);
		return updatedBasket;
	}

	/**
	 * Remove an estimation from the basket by index
	 * @param {number} index - Index of the estimation to remove
	 * @returns {Array} Updated basket array
	 */
	static removeFromBasket(index) {
		const basket = this.loadBasket();
		const updatedBasket = [...basket.slice(0, index), ...basket.slice(index + 1)];
		this.saveBasket(updatedBasket);
		return updatedBasket;
	}

	/**
	 * Clear the entire basket
	 * @returns {Array} Empty array
	 */
	static clearBasket() {
		this.saveBasket([]);
		return [];
	}

	/**
	 * Calculate total QPU seconds for all items in the basket
	 * @returns {number} Total QPU seconds
	 */
	static getTotalQPUSeconds() {
		const basket = this.loadBasket();
		return basket.reduce((total, item) => total + parseFloat(item.qpuSeconds), 0);
	}
}

export default JobHistoryManager;