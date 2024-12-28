<script setup lang="ts">

defineProps<{winner: string, confidence: number, advantages: []}>()

</script>

<template>
  <div>
  <h2 class="text-2xl font-bold text-center text-gray-800">Prediction Results</h2>
  <p class="mt-4 text-gray-700">Winner: <span class="font-bold">{{ winner }} </span></p>
  <p class="mt-2 text-gray-700">Confidence: <span class="font-bold">{{ confidence }}</span></p>
  <div class="mt-4">
    <h3 class="font-semibold text-gray-800">Advantages:</h3>
    <ul class="list-disc list-inside mt-2">
      <li v-for="(value, key) in advantages" :key="key">
        {{ key }}: {{ value }}
      </li>
    </ul>
  </div>
  </div>
</template>