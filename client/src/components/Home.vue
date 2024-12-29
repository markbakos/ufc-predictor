<script setup lang="ts">
import {ref, onMounted} from "vue";
import axios from "axios";
import Prediction from "./Prediction.vue";

interface Results {
  winner: string,
  confidence: number,
  advantages: Record<string, number>
}

const fighter1 = ref('')
const fighter2 = ref('')
const isLoading = ref<boolean>(false)
const predictionResult = ref<Results | null>(null)
const showPrediction = ref(false)

const predictFight = async () => {
  if(isLoading.value) return

  if(!fighter1.value || !fighter2.value) {
    alert("Please enter both fighter's name")
    return
  }

  isLoading.value = true
  showPrediction.value = false

  try{
    const response = await axios.post('http://127.0.0.1:8000/predict/', {
      fighter1: fighter1.value,
      fighter2: fighter2.value
    })

    predictionResult.value = {
      winner: response.data.winner,
      confidence: response.data.confidence,
      advantages: response.data.advantages
    }

    showPrediction.value = true
  }
  catch (error) {
    if (axios.isAxiosError(error)) {
      console.error("Error:", error.response?.data || error.message)
    } else {
      console.error("Error:", error)
    }
  }
  finally{
    isLoading.value = false
  }
}

onMounted(() => {
  document.querySelector('.title')?.classList.add('animate-title')
})
</script>

<template>
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900 p-6">
    <div class="w-full max-w-4xl rounded-xl shadow-2xl overflow-hidden bg-gray-700">
      <div class="p-8">
        <h1 class="title text-5xl font-bold text-center mb-12 text-gray-100 opacity-0 transition-opacity duration-1000">UFC Fight Predictor</h1>
        <div class="flex flex-col md:flex-row justify-between space-y-6 md:space-y-0 md:space-x-6">
          <div class="flex-1 transform hover:scale-105 transition-transform duration-300">
            <h2 class="text-2xl font-semibold mb-3 text-center text-gray-300">Fighter 1</h2>
            <input
                type="text"
                class="w-full px-4 py-3 bg-gray-800 border-2 border-gray-600 rounded-lg text-center text-gray-100 focus:border-gray-500 focus:ring focus:ring-gray-500 focus:ring-opacity-50 transition-all duration-300"
                maxlength="30"
                v-model="fighter1"
                placeholder="Enter fighter name"
            />
          </div>
          <div class="flex-1 transform hover:scale-105 transition-transform duration-300">
            <h2 class="text-2xl font-semibold mb-3 text-center text-gray-300">Fighter 2</h2>
            <input
                type="text"
                class="w-full px-4 py-3 bg-gray-800 border-2 border-gray-600 rounded-lg text-center text-gray-100 focus:border-gray-500 focus:ring focus:ring-gray-500 focus:ring-opacity-50 transition-all duration-300"
                maxlength="30"
                v-model="fighter2"
                placeholder="Enter fighter name"
            />
          </div>
        </div>
        <div class="mt-12 flex justify-center">
          <button
              @click="predictFight"
              :disabled="isLoading"
              class="px-8 py-4 bg-gray-800 text-gray-100 font-bold text-xl rounded-lg shadow-lg hover:bg-gray-600 transform hover:scale-105 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500"
          >
            <span v-if="!isLoading">Predict Fight</span>
            <span v-else class="flex items-center">
              <svg class="animate-spin -ml-1 mr-3 h-5 w-5 text-gray-100" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Predicting...
            </span>
          </button>
        </div>
      </div>
      <div v-if="predictionResult" class="p-8" :class="{ 'opacity-100 transition-opacity duration-500': showPrediction, 'opacity-0': !showPrediction }">
        <Prediction
            :winner="predictionResult.winner"
            :confidence="predictionResult.confidence"
            :advantages="predictionResult.advantages"
        />
      </div>
    </div>
  </div>
</template>

<style scoped>
@keyframes titleAnimation {
  0% { opacity: 0; transform: translateY(-20px); }
  100% { opacity: 1; transform: translateY(0); }
}

.animate-title {
  animation: titleAnimation 1s ease-out forwards;
}
</style>
