import { initializeApp } from "firebase/app";
import {getFirestore} from "firebase/firestore"

const firebaseConfig = {
  apiKey: "AIzaSyCUQ4elH8AJZSfxucRKPPwUFlT3GZDWlm0",
  authDomain: "rebot-dd897.firebaseapp.com",
  projectId: "rebot-dd897",
  storageBucket: "rebot-dd897.firebasestorage.app",
  messagingSenderId: "367251853211",
  appId: "1:367251853211:web:c01c59fcc92c4671429b3e"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore();