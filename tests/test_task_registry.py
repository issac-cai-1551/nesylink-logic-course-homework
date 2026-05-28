import unittest


class TaskRegistryTests(unittest.TestCase):
    def test_builtin_tasks_are_registered(self):
        from nesylink.tasks import get_task, list_tasks

        task_ids = [task.task_id for task in list_tasks()]
        self.assertIn("collect_key_easy", task_ids)
        self.assertIn("kill_monsters_easy", task_ids)
        self.assertIn("avoid_traps_easy", task_ids)

        task = get_task("collect_key_easy")
        self.assertEqual(task.map_id, "key_door")
        self.assertEqual(task.reward_id, "collect_key")
        self.assertEqual(task.gym_id, "NesyLink-CollectKeyEasy-v0")

    def test_unknown_task_id_has_clear_error(self):
        from nesylink.tasks import get_task

        with self.assertRaisesRegex(ValueError, "unknown task_id 'missing'"):
            get_task("missing")


if __name__ == "__main__":
    unittest.main()
