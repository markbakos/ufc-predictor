<script setup lang="ts">
import {ref} from "vue";
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

const predictFight = async () => {

  if(isLoading.value) return

  if(!fighter1.value || !fighter2.value) {
    alert("Please enter both fighter's name")
    return
  }

  isLoading.value = true

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

    console.log(predictionResult)

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
</script>

<template>
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-700 to-gray-900 p-6">
    <div class="w-full max-w-4xl bg-white rounded-xl shadow-2xl overflow-hidden">
      <div class="p-8">
        <h1 class="text-4xl font-bold text-center mb-8 text-gray-800">Predict Fight</h1>
        <div class="flex flex-col md:flex-row justify-between space-y-6 md:space-y-0 md:space-x-6">
          <div class="flex-1">
            <h2 class="text-xl font-semibold mb-2 text-center text-gray-700">Fighter 1</h2>
            <input
                type="text"
                class="w-full px-4 py-2 border border-gray-300 rounded-lg text-center"
                maxlength="30"
                v-model="fighter1"
                placeholder="Enter fighter name"
            />
          </div>
          <div class="flex-1">
            <h2 class="text-xl font-semibold mb-2 text-center text-gray-700">Fighter 2</h2>
            <input
                type="text"
                class="w-full px-4 py-2 border border-gray-300 rounded-lg text-center"
                maxlength="30"
                v-model="fighter2"
                placeholder="Enter fighter name"
            />
          </div>
        </div>
        <div class="mt-8 flex justify-center">
          <button
              @click="predictFight"
              class="px-6 py-3 bg-white border-[1.5px] border-black text-black font-semibold rounded-lg shadow-md flex items-center"
          >
            Predict
          </button>
        </div>
      </div>
      <div v-if="predictionResult" class="p-8 bg-gray-100">
        <Prediction
            :winner="predictionResult.winner"
            :confidence="predictionResult.confidence"
            :advantages="predictionResult.advantages"
        />
      </div>
    </div>
  </div>
</template>