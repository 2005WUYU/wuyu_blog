import { defineCollection, z } from 'astro:content';

const blogCollection = defineCollection({
    schema: z.object({
        title: z.string(),
        tags: z.array(z.string()),
        image: z.string().optional(),
        pubDate: z.date(),
    })
});

export const collections = {
    'study': blogCollection,
    'essay': blogCollection,
    'tech': blogCollection,
};
